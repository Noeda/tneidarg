-- Bare bones tensor wrapping around some Cuda and cuDNN

module Tensor
  ( Tensor()
  , rows
  , cols
  , touchTensor
  , fromRows
  , toRows
  , zeros
  , viewColumnVec
  , viewColumnVecOffset
  , subtract
  , add
  , sigmoid
  , sigmoidTanh
  , lstmMemory
  , lstmOutput
  , lstmBiasLastAct
  , copy
  , matMul
  , matMulVec
  , matMulBatched
  , matMulBatchedAdd
  , outerProduct )
  where

import Control.Concurrent
import Control.Exception
import Control.Monad
import Control.Monad.Primitive ( touch )
import Data.Foldable
import Data.Traversable
import Numeric.Half
import Foreign.ForeignPtr
import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable
import Prelude hiding ( subtract )
import System.Mem
import System.IO.Unsafe

foreign import ccall "init_cuda" c_init_cuda :: IO (Ptr ())
-- foreign import ccall "shutdown_cuda" c_shutdown_cuda :: Ptr () -> IO ()
foreign import ccall "cuda_alloc_2d" c_cuda_alloc_2d :: CSize -> CSize -> CSize -> Ptr (Ptr ()) -> Ptr CSize -> IO CInt
foreign import ccall "cuda_memset_2d" c_cuda_memset_2d :: Ptr () -> CSize -> CSize -> CSize -> CInt -> IO ()
foreign import ccall "cuda_copy_from_host_to_device_2d" c_cuda_copy_from_host_to_device_2d :: Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "cuda_copy_from_device_to_host_2d" c_cuda_copy_from_device_to_host_2d :: Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "cuda_copy_from_device_to_device_2d" c_cuda_copy_from_device_to_device_2d :: Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "&cuda_dealloc" c_ptr_cuda_dealloc :: FunPtr (Ptr () -> IO ())
foreign import ccall "cuda_matmul" c_cuda_matmul ::
  Ptr () -> Ptr () -> CSize -> Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "cuda_matmul_vec" c_cuda_matmul_vec ::
  Ptr () -> Ptr () -> CSize -> Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "cuda_matmul_batched" c_cuda_matmul_batched ::
  Ptr () -> Ptr (Ptr ()) -> CSize -> Ptr (Ptr ()) -> CSize -> Ptr (Ptr ()) -> CSize -> CSize -> CSize -> CSize -> CInt -> CDouble -> IO ()
foreign import ccall "cuda_sigmoid" c_cuda_sigmoid ::
  Ptr () -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "cuda_sigmoid_tanh" c_cuda_sigmoid_tanh ::
  Ptr () -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "cuda_lstm_memory" c_cuda_lstm_memory ::
  Ptr () -> Ptr () -> Ptr () -> Ptr () -> Ptr () -> CSize -> IO ()
foreign import ccall "cuda_lstm_output" c_cuda_lstm_output ::
  Ptr () -> Ptr () -> Ptr () -> CSize -> IO ()
foreign import ccall "cuda_lstm_bias_last_act" c_cuda_lstm_bias_last_act ::
  Ptr () -> Ptr () -> Ptr () -> Ptr () -> CSize -> IO ()
foreign import ccall "cuda_sub" c_cuda_sub ::
  Ptr () -> CSize -> Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "cuda_add" c_cuda_add ::
  Ptr () -> CSize -> Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "cuda_outer_product" c_cuda_outer_product ::
  Ptr () -> Ptr () -> CSize -> Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> IO ()

-- width of a float16
dtWidth :: Integral a => a
dtWidth = 2

{-# NOINLINE cudaInitialized #-}
cudaInitialized :: MVar (Ptr ())
cudaInitialized = unsafePerformIO $ newMVar nullPtr

initCuda :: IO (Ptr ())
initCuda = modifyMVar cudaInitialized $ \ptr ->
  if ptr == nullPtr
    then do
      new_ptr <- c_init_cuda
      return (new_ptr, new_ptr)
    else return (ptr, ptr)

data Tensor = Tensor
  { rawTensor :: !(ForeignPtr ())
  , tensorPitch :: !Int
  , tensorRows :: !Int
  , tensorCols :: !Int }

viewColumnVec :: Tensor -> Int -> Tensor
viewColumnVec vec _ | tensorCols vec /= 1 = error "viewColumnVec: tensorCols vec /= 1"
viewColumnVec vec new_rows | new_rows == tensorRows vec = vec
viewColumnVec vec new_rows | new_rows > tensorRows vec =
  error "viewColumnVec: new_rows > tensorRows vec"
viewColumnVec _ 0 = error "viewColumnVec: new_row == 0"
viewColumnVec vec new_rows = vec { tensorRows = new_rows }

-- viewColumnVecOffset tensor sz offset
viewColumnVecOffset :: Tensor -> Int -> Int -> Tensor
viewColumnVecOffset tensor _ _ | tensorCols tensor /= 1 = error "viewColumnVecOffset: tensorCols tensor /= 1"
viewColumnVecOffset tensor sz 0 | sz == tensorRows tensor = tensor
viewColumnVecOffset _ 0 _ = error "viewColumnVecOffset: new_row == 0"
viewColumnVecOffset vec new_rows offset =
  if new_rows > new_unviewed_sz
    then error "viewColumnVecOffset: new_rows > new_unviewed_sz"
    else vec { tensorRows = new_rows
             , rawTensor = new_ptr }
 where
  new_unviewed_sz = tensorRows vec - offset
  new_ptr = plusForeignPtr (rawTensor vec) (offset * dtWidth)

rows :: Tensor -> Int
rows = tensorRows

cols :: Tensor -> Int
cols = tensorCols

touchTensor :: Tensor -> IO ()
touchTensor tensor = withTensorPtr tensor $ \ptr ->
  touch ptr

withTensorPtr :: Tensor -> (Ptr () -> IO a) -> IO a
withTensorPtr tensor f = withForeignPtr (rawTensor tensor) f

withTensorPtrs :: [Tensor] -> (Ptr (Ptr ()) -> IO a) -> IO a
withTensorPtrs [] f = f nullPtr
withTensorPtrs tensors f = allocaArray (length tensors) $ \ptrs ->
  go ptrs tensors 0
 where
  go ptrs (tensor:rest) idx =
    withTensorPtr tensor $ \ptr -> do
      pokeElemOff ptrs idx ptr
      go ptrs rest (idx+1)
  go ptrs _ _ = f ptrs

instance Show Tensor where
  show tensor = "Tensor<" ++ show (tensorRows tensor) ++ "x" ++ show (tensorCols tensor) ++ " ptr: " ++ show (rawTensor tensor) ++ ">"

allocate :: Int -> Int -> IO Tensor
allocate rows cols = mask_ $ do
  void initCuda
  alloca $ \receiving_ptr -> do
    alloca $ \receiving_pitch -> do
      retval <- c_cuda_alloc_2d (fromIntegral rows)
                                (fromIntegral cols)
                                dtWidth
                                receiving_ptr
                                receiving_pitch
      -- If allocation fails, try GCing and then attempt once more, in the hope
      -- that finalizers have run and freed up some memory.
      when (retval /= 0) $ do
        performGC
        retval2 <- c_cuda_alloc_2d (fromIntegral rows)
                                   (fromIntegral cols)
                                   dtWidth
                                   receiving_ptr
                                   receiving_pitch
        when (retval2 /= 0) $
          error "cuda_alloc_2d failed"
      fptr <- newForeignPtr c_ptr_cuda_dealloc =<< peek receiving_ptr
      pitch <- peek receiving_pitch
      return $ Tensor { rawTensor = fptr
                      , tensorPitch = fromIntegral pitch
                      , tensorRows = rows
                      , tensorCols = cols }

zeros :: Int -> Int -> IO Tensor
zeros rows cols = do
  tensor <- allocate rows cols
  withTensorPtr tensor $ \ptr ->
    c_cuda_memset_2d ptr (fromIntegral (tensorPitch tensor))
                         (fromIntegral (tensorRows tensor) * dtWidth)
                         (fromIntegral (tensorCols tensor))
                         0
  return tensor

{-# INLINE doubleToFloat16 #-}
doubleToFloat16 :: Double -> Half
doubleToFloat16 = fromRational . toRational

{-# INLINE float16ToDouble #-}
float16ToDouble :: Half -> Double
float16ToDouble = fromRational . toRational

fromRows :: [[Double]] -> IO Tensor
fromRows values = do
  let nrows = length values
  when (nrows == 0) $
    error "Cannot create a tensor from an empty list of rows"
  let ncols = length (head values)
  when (ncols == 0) $
    error "Cannot create a tensor from an empty row"
  unless (all ((== ncols) . length) values) $
    error "Cannot create a tensor from a jagged list of rows"
  tensor <- allocate nrows ncols
  allocaArray (nrows*ncols) $ \(!ptr :: Ptr Half) -> do
    for_ (zip [0..] values) $ \(!row, !row_values) -> do
      for_ (zip [0..] row_values) $ \(!col, !value) -> do
        pokeElemOff ptr (row + col * nrows) (doubleToFloat16 value)
    withTensorPtr tensor $ \tensor_ptr ->
      c_cuda_copy_from_host_to_device_2d tensor_ptr
                                         (fromIntegral (tensorPitch tensor))
                                         (castPtr ptr)
                                         (fromIntegral (tensorRows tensor) * dtWidth)
                                         (fromIntegral (tensorRows tensor) * dtWidth)
                                         (fromIntegral (tensorCols tensor))
  return tensor

toRows :: Tensor -> IO [[Double]]
toRows tensor = do
  allocaArray (tensorRows tensor * tensorCols tensor) $ \(!ptr :: Ptr Half) -> do
    withTensorPtr tensor $ \tensor_ptr ->
      c_cuda_copy_from_device_to_host_2d (castPtr ptr)
                                         (fromIntegral (tensorRows tensor) * dtWidth)
                                         tensor_ptr
                                         (fromIntegral (tensorPitch tensor))
                                         (fromIntegral (tensorRows tensor) * dtWidth)
                                         (fromIntegral (tensorCols tensor))
    for [0..tensorRows tensor - 1] $ \row -> do
      for [0..tensorCols tensor - 1] $ \col -> do
        peekElemOff ptr (row + col * tensorRows tensor) >>= return . float16ToDouble

-- copies contents of one tensor to another. Tensor dimensions must match.
-- copy dst src
copy :: Tensor -> Tensor -> IO ()
copy dst src = do
  when (tensorRows dst /= tensorRows src) $
    error "Destination tensor has incompatible dimensions"
  when (tensorCols dst /= tensorCols src) $
    error "Destination tensor has incompatible dimensions"
  withTensorPtr dst $ \dst_ptr ->
    withTensorPtr src $ \src_ptr ->
      c_cuda_copy_from_device_to_device_2d
        dst_ptr
        (fromIntegral $ tensorPitch dst)
        src_ptr
        (fromIntegral $ tensorPitch src)
        (fromIntegral $ tensorRows dst)
        (fromIntegral $ tensorCols dst)

-- subtract two matrices
subtract :: Tensor -> Tensor -> Tensor -> IO ()
subtract dst mat1 mat2 = do
  void initCuda
  when (tensorRows mat1 /= tensorRows mat2) $
    error "Cannot subtract matrices with incompatible dimensions"
  when (tensorCols mat1 /= tensorCols mat2) $
    error "Cannot subtract matrices with incompatible dimensions"
  when (tensorRows dst /= tensorRows mat1) $
    error "Destination matrix has incompatible dimensions"
  when (tensorCols dst /= tensorCols mat1) $
    error "Destination matrix has incompatible dimensions"
  withTensorPtr dst $ \dst_ptr ->
    withTensorPtr mat1 $ \mat1_ptr ->
      withTensorPtr mat2 $ \mat2_ptr ->
        c_cuda_sub dst_ptr (fromIntegral (tensorPitch dst))
                   mat1_ptr (fromIntegral (tensorPitch mat1))
                   mat2_ptr (fromIntegral (tensorPitch mat2))
                   (fromIntegral $ tensorRows dst)
                   (fromIntegral $ tensorCols dst)

-- subtract two matrices
add :: Tensor -> Tensor -> Tensor -> IO ()
add dst mat1 mat2 = do
  void initCuda
  when (tensorRows mat1 /= tensorRows mat2) $
    error "Cannot add matrices with incompatible dimensions"
  when (tensorCols mat1 /= tensorCols mat2) $
    error "Cannot add matrices with incompatible dimensions"
  when (tensorRows dst /= tensorRows mat1) $
    error "Destination matrix has incompatible dimensions"
  when (tensorCols dst /= tensorCols mat1) $
    error "Destination matrix has incompatible dimensions"
  withTensorPtr dst $ \dst_ptr ->
    withTensorPtr mat1 $ \mat1_ptr ->
      withTensorPtr mat2 $ \mat2_ptr ->
        c_cuda_add dst_ptr (fromIntegral (tensorPitch dst))
                   mat1_ptr (fromIntegral (tensorPitch mat1))
                   mat2_ptr (fromIntegral (tensorPitch mat2))
                   (fromIntegral $ tensorRows dst)
                   (fromIntegral $ tensorCols dst)

outerProduct :: Tensor -> Tensor -> Tensor -> IO ()
outerProduct dst vec1 vec2 = do
  cublas_ptr <- initCuda
  when (tensorCols vec1 /= 1) $
    error "Cannot compute outer product of a non-vector"
  when (tensorCols vec2 /= 1) $
    error "Cannot compute outer product of a non-vector"
  when (tensorRows dst /= tensorRows vec1) $
    error "Destination matrix has incompatible dimensions"
  when (tensorCols dst /= tensorRows vec2) $
    error "Destination matrix has incompatible dimensions"

  withTensorPtr dst $ \dst_ptr ->
    withTensorPtr vec1 $ \vec1_ptr ->
      withTensorPtr vec2 $ \vec2_ptr ->
        c_cuda_outer_product cublas_ptr
                             dst_ptr (fromIntegral (tensorPitch dst))
                             vec1_ptr (fromIntegral (tensorPitch vec1))
                             vec2_ptr (fromIntegral (tensorPitch vec2))
                             (fromIntegral $ tensorRows dst)
                             (fromIntegral $ tensorCols dst)

-- dst mat1 mat2
matMul :: Tensor -> Tensor -> Tensor -> IO ()
matMul dst mat1 mat2 = do
  cublas_ptr <- initCuda
  when (tensorCols mat1 /= tensorRows mat2) $
    error "Cannot multiply matrices with incompatible dimensions"
  when (tensorRows dst /= tensorRows mat1) $
    error "Destination matrix has incompatible dimensions"
  when (tensorCols dst /= tensorCols mat2) $
    error "Destination matrix has incompatible dimensions"
  withTensorPtr dst $ \dst_ptr ->
    withTensorPtr mat1 $ \mat1_ptr ->
      withTensorPtr mat2 $ \mat2_ptr ->
        c_cuda_matmul cublas_ptr
                      dst_ptr (fromIntegral (tensorPitch dst))
                      mat1_ptr (fromIntegral (tensorPitch mat1))
                      mat2_ptr (fromIntegral (tensorPitch mat2))
                      (fromIntegral $ tensorRows dst)
                      (fromIntegral $ tensorCols dst)
                      (fromIntegral $ tensorCols mat1)

-- matrix-vector multiply, third tensor is expected to be a column vector
-- dst mat1 vec2.
--
-- result will be given as another column vector
matMulVec :: Tensor -> Tensor -> Tensor -> IO ()
matMulVec dst mat1 vec2 = do
  cublas_ptr <- initCuda
  when (tensorCols mat1 /= tensorRows vec2) $
    error "Cannot multiply matrices with incompatible dimensions"
  when (tensorRows dst /= tensorRows mat1) $
    error "Destination matrix has incompatible dimensions"
  when (tensorCols dst /= 1) $
    error "Destination matrix has incompatible dimensions (1 column expected)"
  withTensorPtr dst $ \dst_ptr ->
    withTensorPtr mat1 $ \mat1_ptr ->
      withTensorPtr vec2 $ \vec2_ptr ->
        c_cuda_matmul_vec cublas_ptr
                          dst_ptr (fromIntegral (tensorPitch dst))
                          mat1_ptr (fromIntegral (tensorPitch mat1))
                          vec2_ptr (fromIntegral (tensorPitch vec2))
                          (fromIntegral $ tensorRows dst)
                          (fromIntegral $ tensorCols mat1)

matMulBatched :: [Tensor] -> [Tensor] -> [Tensor] -> IO ()
matMulBatched dsts mat1s mat2s = matMulBatchedAdd dsts mat1s mat2s 0

matMulBatchedAdd :: [Tensor] -> [Tensor] -> [Tensor] -> Double -> IO ()
matMulBatchedAdd dsts mat1s mat2s multiplier = do
  cublas_ptr <- initCuda
  let nbatches = length dsts
  when (length mat1s /= nbatches) $
    error "Cannot multiply matrices with incompatible dimensions"
  when (length mat2s /= nbatches) $
    error "Cannot multiply matrices with incompatible dimensions"
  for_ (zip3 dsts mat1s mat2s) $ \(dst, mat1, mat2) -> do
    when (tensorCols mat1 /= tensorRows mat2) $
      error "Cannot multiply matrices with incompatible dimensions"
    when (tensorRows dst /= tensorRows mat1) $
      error "Destination matrix has incompatible dimensions"
    when (tensorCols dst /= tensorCols mat2) $
      error "Destination matrix has incompatible dimensions"


  withTensorPtrs dsts $ \dst_ptrs ->
    withTensorPtrs mat1s $ \mat1_ptrs ->
      withTensorPtrs mat2s $ \mat2_ptrs ->
        c_cuda_matmul_batched cublas_ptr
                              dst_ptrs (fromIntegral (tensorPitch (head dsts)))
                              mat1_ptrs (fromIntegral (tensorPitch (head mat1s)))
                              mat2_ptrs (fromIntegral (tensorPitch (head mat2s)))
                              (fromIntegral $ tensorRows (head dsts))
                              (fromIntegral $ tensorCols (head dsts))
                              (fromIntegral $ tensorCols (head mat1s))
                              (fromIntegral nbatches)
                              (CDouble multiplier)

-- Apply sigmoid function to a tensor
sigmoid :: Tensor -> IO ()
sigmoid tensor = do
  withTensorPtr tensor $ \tensor_ptr ->
    c_cuda_sigmoid tensor_ptr
                   (fromIntegral $ tensorPitch tensor)
                   (fromIntegral $ tensorRows tensor)
                   (fromIntegral $ tensorCols tensor)

sigmoidTanh :: Tensor -> IO ()
sigmoidTanh tensor = do
  withTensorPtr tensor $ \tensor_ptr ->
    c_cuda_sigmoid_tanh tensor_ptr
                        (fromIntegral $ tensorPitch tensor)
                        (fromIntegral $ tensorRows tensor)
                        (fromIntegral $ tensorCols tensor)

-- Applies a calculation used in LSTM:
-- new_memory = memory * forget_gate + input_gate * input
-- lstmMemory new_memory memory forget_gate input_gate input
lstmMemory :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> IO ()
lstmMemory new_memory memory forget_gate input_gate input = do
  when (tensorCols new_memory /= 1) $
    error "Destination matrix has incompatible dimensions (1 column expected)"
  when (tensorCols memory /= 1) $
    error "Source memory matrix has incompatible dimensions (1 column expected)"
  when (tensorCols forget_gate /= 1) $
    error "Forget gate matrix has incompatible dimensions (1 column expected)"
  when (tensorCols input_gate /= 1) $
    error "Input gate matrix has incompatible dimensions (1 column expected)"
  when (tensorCols input /= 1) $
    error "Input matrix has incompatible dimensions (1 column expected)"
  when (tensorRows new_memory /= tensorRows memory) $
    error "Destination matrix has incompatible dimensions"
  when (tensorRows new_memory /= tensorRows forget_gate) $
    error "Destination matrix has incompatible dimensions"
  when (tensorRows new_memory /= tensorRows input_gate) $
    error "Destination matrix has incompatible dimensions"
  when (tensorRows new_memory /= tensorRows input) $
    error "Destination matrix has incompatible dimensions"

  withTensorPtr new_memory $ \new_memory_ptr ->
    withTensorPtr memory $ \memory_ptr ->
      withTensorPtr forget_gate $ \forget_gate_ptr ->
        withTensorPtr input_gate $ \input_gate_ptr ->
          withTensorPtr input $ \input_ptr ->
            c_cuda_lstm_memory new_memory_ptr
                               memory_ptr
                               forget_gate_ptr
                               input_gate_ptr
                               input_ptr
                               (fromIntegral $ tensorRows new_memory)

-- computes:
-- out = tanh(x) * sigmoid(y)
-- lstmOutput out x y
lstmOutput :: Tensor -> Tensor -> Tensor -> IO ()
lstmOutput out x y = do
  when (tensorCols out /= 1) $
    error "Destination matrix has incompatible dimensions (1 column expected)"
  when (tensorCols x /= 1) $
    error "Source x matrix has incompatible dimensions (1 column expected)"
  when (tensorCols y /= 1) $
    error "Source y matrix has incompatible dimensions (1 column expected)"
  when (tensorRows out /= tensorRows x) $
    error "Destination matrix has incompatible dimensions"
  when (tensorRows out /= tensorRows y) $
    error "Destination matrix has incompatible dimensions"

  withTensorPtr out $ \out_ptr ->
    withTensorPtr x $ \x_ptr ->
      withTensorPtr y $ \y_ptr ->
        c_cuda_lstm_output out_ptr
                           x_ptr
                           y_ptr
                           (fromIntegral $ tensorRows out)

-- computes:
-- out = bias + weight * act
-- lstmBiasLastAct out bias weight act
lstmBiasLastAct :: Tensor -> Tensor -> Tensor -> Tensor -> IO ()
lstmBiasLastAct out bias weight act = do
  when (tensorCols out /= 1) $
    error "Destination matrix has incompatible dimensions (1 column expected)"
  when (tensorCols bias /= 1) $
    error "Source x matrix has incompatible dimensions (1 column expected)"
  when (tensorCols weight /= 1) $
    error "Source y matrix has incompatible dimensions (1 column expected)"
  when (tensorCols act /= 1) $
    error "Source y matrix has incompatible dimensions (1 column expected)"
  when (tensorRows out /= tensorRows bias) $
    error "Destination matrix has incompatible dimensions"
  when (tensorRows out /= tensorRows weight) $
    error "Destination matrix has incompatible dimensions"
  when (tensorRows out /= tensorRows act) $
    error "Destination matrix has incompatible dimensions"

  withTensorPtr out $ \out_ptr ->
    withTensorPtr bias $ \bias_ptr ->
      withTensorPtr weight $ \weight_ptr ->
        withTensorPtr act $ \act_ptr ->
          c_cuda_lstm_bias_last_act out_ptr
                                    bias_ptr
                                    weight_ptr
                                    act_ptr
                                    (fromIntegral $ tensorRows out)
