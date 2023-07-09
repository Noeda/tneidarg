-- Bare bones tensor wrapping around some Cuda and cuDNN

module Tensor
  ( Tensor()
  , rows
  , cols
  , touchTensor
  , fromRows
  , toRows
  , fromRowsVec
  , toRowsVec
  , zeros
  , Seed
  , gaussian
  , view
  , subtract
  , add
  , scale
  , sigmoid
  , sigmoidTanh
  , copy
  , matMul
  , matMulVec
  , matMulBatched
  , matMulBatchedAdd
  , outerProduct
  -- * Streams and events
  , Stream()
  , makeStream
  , nullStream
  , Event()
  , makeEvent
  , makeStreamWaitForEvent )
  where

import Control.Concurrent
import Control.Exception
import Control.Monad
import Control.Monad.Primitive ( touch )
import Data.Foldable
import Data.Int
import Data.Traversable
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VU
import Numeric.Half
import Foreign.ForeignPtr
import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Marshal.Utils
import Foreign.Ptr
import Foreign.Storable
import Prelude hiding ( subtract )
import System.Mem
import System.IO.Unsafe
import Test.QuickCheck hiding ( scale )

foreign import ccall "init_cuda" c_init_cuda :: IO (Ptr ())
-- foreign import ccall "shutdown_cuda" c_shutdown_cuda :: Ptr () -> IO ()
foreign import ccall "cuda_alloc_2d" c_cuda_alloc_2d :: CSize -> CSize -> CSize -> Ptr (Ptr ()) -> Ptr CSize -> IO CInt
foreign import ccall "cuda_memset_2d" c_cuda_memset_2d :: Ptr () -> CSize -> CSize -> CSize -> CInt -> Ptr () -> IO ()
foreign import ccall "cuda_gaussian_random_2d" c_cuda_gaussian_random_2d :: Ptr () -> CInt -> CSize -> CSize -> CSize -> CInt -> Ptr () -> CDouble -> CDouble -> IO ()
foreign import ccall "cuda_copy_from_host_to_device_2d" c_cuda_copy_from_host_to_device_2d :: Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "cuda_copy_from_device_to_host_2d" c_cuda_copy_from_device_to_host_2d :: Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> IO ()
foreign import ccall "cuda_copy_from_device_to_device_2d" c_cuda_copy_from_device_to_device_2d :: Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> Ptr () -> IO ()
foreign import ccall "&cuda_dealloc" c_ptr_cuda_dealloc :: FunPtr (Ptr () -> IO ())
foreign import ccall "cuda_matmul" c_cuda_matmul ::
  Ptr () -> Ptr () -> CSize -> Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> CSize -> Ptr () -> IO ()
foreign import ccall "cuda_matmul_vec" c_cuda_matmul_vec ::
  Ptr () -> Ptr () -> CSize -> Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> Ptr () -> IO ()
foreign import ccall "cuda_matmul_batched" c_cuda_matmul_batched ::
  Ptr () -> Ptr (Ptr ()) -> CSize -> Ptr (Ptr ()) -> CSize -> Ptr (Ptr ()) -> CSize -> CSize -> CSize -> CSize -> CInt -> CDouble -> Ptr () -> IO ()
foreign import ccall "cuda_sigmoid" c_cuda_sigmoid ::
  Ptr () -> CSize -> CSize -> CSize -> Ptr () -> IO ()
foreign import ccall "cuda_sigmoid_tanh" c_cuda_sigmoid_tanh ::
  Ptr () -> CSize -> CSize -> CSize -> Ptr () -> IO ()
foreign import ccall "cuda_sub" c_cuda_sub ::
  Ptr () -> CSize -> Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> Ptr () -> IO ()
foreign import ccall "cuda_add" c_cuda_add ::
  Ptr () -> CSize -> Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> Ptr () -> IO ()
foreign import ccall "cuda_outer_product" c_cuda_outer_product ::
  Ptr () -> Ptr () -> CSize -> Ptr () -> CSize -> Ptr () -> CSize -> CSize -> CSize -> Ptr () -> IO ()
foreign import ccall "cuda_scale" c_cuda_scale ::
  Ptr () -> Ptr () -> CSize -> CSize -> CSize -> Ptr () -> IO ()
foreign import ccall unsafe "cuda_size_of_cuda_event_t" c_cuda_size_of_cuda_event_t :: CSize
foreign import ccall "cuda_create_event" c_cuda_create_event :: Ptr () -> Ptr () -> IO ()
foreign import ccall "&cuda_destroy_event" c_cuda_destroy_event :: FunPtr (Ptr () -> IO ())
foreign import ccall unsafe "cuda_size_of_cuda_stream_t" c_cuda_size_of_cuda_stream_t :: CSize
foreign import ccall "cuda_create_stream" c_cuda_create_stream :: Ptr () -> IO ()
foreign import ccall "&cuda_destroy_stream" c_cuda_destroy_stream :: FunPtr (Ptr () -> IO ())
foreign import ccall "cuda_make_stream_wait_for_event" c_cuda_make_stream_wait_for_event :: Ptr () -> Ptr () -> IO ()

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

-- cublas does not like if you use the same handle concurrently from multiple
-- threads.
{-# NOINLINE blasLock #-}
blasLock :: MVar ()
blasLock = unsafePerformIO $ newMVar ()

{-# INLINE withBlasLock #-}
withBlasLock :: IO a -> IO a
withBlasLock action = withMVar blasLock $ \_ -> action

---------
-- Events
---------
data Event = Event
  -- cudaEvent_t*
  { eventPtr :: !(ForeignPtr ()) }
  deriving ( Eq, Ord )

makeEvent :: Stream -> IO Event
makeEvent (Stream stream_fptr) = mask_ $ do
  fptr <- mallocForeignPtrBytes (fromIntegral c_cuda_size_of_cuda_event_t)
  withForeignPtr fptr $ \ptr ->
    withForeignPtr stream_fptr $ \stream_ptr ->
      c_cuda_create_event ptr stream_ptr
  addForeignPtrFinalizer c_cuda_destroy_event fptr
  return $ Event { eventPtr = fptr }

-----------
-- Stream
-----------
data Stream = Stream
  { streamPtr :: !(ForeignPtr ()) }
  deriving ( Eq, Ord )

makeStream :: IO Stream
makeStream = mask_ $ do
  fptr <- mallocForeignPtrBytes (fromIntegral c_cuda_size_of_cuda_stream_t)
  withForeignPtr fptr $ \ptr ->
    c_cuda_create_stream ptr
  addForeignPtrFinalizer c_cuda_destroy_stream fptr
  return $ Stream { streamPtr = fptr }

{-# NOINLINE nullStream #-}
nullStream :: Stream
nullStream = unsafePerformIO $ mask_ $ do
  fptr <- mallocForeignPtrBytes (fromIntegral c_cuda_size_of_cuda_stream_t)
  withForeignPtr fptr $ \ptr ->
    fillBytes ptr 0 (fromIntegral c_cuda_size_of_cuda_stream_t)
  return $ Stream { streamPtr = fptr }

makeStreamWaitForEvent :: Stream -> Event -> IO ()
makeStreamWaitForEvent (Stream stream_fptr) (Event event_fptr) = mask_ $ do
  withForeignPtr stream_fptr $ \stream_ptr ->
    withForeignPtr event_fptr $ \event_ptr ->
      c_cuda_make_stream_wait_for_event stream_ptr event_ptr

data Tensor = Tensor
  { rawTensor :: !(ForeignPtr ())
  , tensorPitch :: !Int
  , tensorRows :: !Int
  , tensorCols :: !Int
  -- if the tensor is asynchronously waiting on an operation to finish,
  -- this will be set to the event that will be signaled when the operation
  -- is complete.
  }

instance Arbitrary Tensor where
  {-# NOINLINE arbitrary #-}
  arbitrary = do
    nrows <- choose (1, 100)
    ncols <- choose (1, 100)
    initial_values <- replicateM nrows $ replicateM ncols $ choose (-1, 1) :: Gen [[Double]]
    -- Concede and use unsafePerformIO to make testing a bit easier.
    -- Can use them in QuickCheck properties as inputs without hassle.
    let !s = unsafePerformIO $ fromRows initial_values
    assert (rows s == nrows) $ assert (cols s == ncols) $ return s

  shrink tensor =
    (if rows tensor > 1
       then [view tensor 0 0 (rows tensor `div` 2) (cols tensor)
            ,view tensor (rows tensor `div` 2) 0 (rows tensor - (rows tensor `div` 2)) (cols tensor)]
       else []) <>
    (if cols tensor > 1
       then [view tensor 0 0 (rows tensor) (cols tensor `div` 2)
            ,view tensor 0 (cols tensor `div` 2) (rows tensor) (cols tensor - (cols tensor `div` 2))]
       else []) <>
    (if rows tensor > 1 && cols tensor > 1
       -- one shrink for each quadrant
       then let half_rows = rows tensor `div` 2
                half_cols = cols tensor `div` 2
            in [view tensor 0 0 half_rows half_cols
               ,view tensor 0 half_cols half_rows (cols tensor - half_cols)
               ,view tensor half_rows 0 (rows tensor - half_rows) half_cols
               ,view tensor half_rows half_cols (rows tensor - half_rows) (cols tensor - half_cols)]
       else [])

-- view tensor row col nrows ncols
view :: Tensor -> Int -> Int -> Int -> Int -> Tensor
view tensor row col nrows ncols =
  if nrows == 0 || ncols == 0
    then error "view: nrows == 0 || ncols == 0"
    else if nrows < 0 || ncols < 0 || row < 0 || col < 0
           then error "view: nrows < 0 || ncols < 0 || row < 0 || col < 0"
           else if row + nrows > tensorRows tensor
                  then error "view: row + nrows > tensorRows tensor"
                  else if col + ncols > tensorCols tensor
                         then error "view: col + ncols > tensorCols tensor"
                         else doTheView
 where
  doTheView = tensor { tensorRows = nrows
                     , tensorCols = ncols
                     , rawTensor = new_ptr }

  offset = col * tensorPitch tensor + row * dtWidth
  new_ptr = plusForeignPtr (rawTensor tensor) offset

rows :: Tensor -> Int
rows = tensorRows

cols :: Tensor -> Int
cols = tensorCols

withStreamPtr :: Stream -> (Ptr () -> IO a) -> IO a
withStreamPtr stream f = withForeignPtr (streamPtr stream) f

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

-- Creates a tensor filled with zeros
zeros :: Stream -> Int -> Int -> IO Tensor
zeros stream rows cols = do
  tensor <- allocate rows cols
  withStreamPtr stream $ \stream_ptr ->
    withTensorPtr tensor $ \ptr ->
      c_cuda_memset_2d ptr (fromIntegral (tensorPitch tensor))
                           (fromIntegral (tensorRows tensor) * dtWidth)
                           (fromIntegral (tensorCols tensor))
                           0
                           stream_ptr
  return tensor

type Seed = Int32

-- Creates a tensor filled with Gaussian distributed values with the given mean
-- and standard deviation.
gaussian :: Stream -> Seed -> Int -> Int -> Double -> Double -> IO Tensor
gaussian stream seed rows cols mean stdev = do
  tensor <- allocate rows cols
  withStreamPtr stream $ \stream_ptr ->
    withTensorPtr tensor $ \ptr ->
      c_cuda_gaussian_random_2d ptr (fromIntegral seed)
                                    (fromIntegral (tensorPitch tensor))
                                    (fromIntegral (tensorRows tensor))
                                    (fromIntegral (tensorCols tensor))
                                    0
                                    stream_ptr
                                    (CDouble mean)
                                    (CDouble stdev)
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

toRowsVec :: Tensor -> IO (V.Vector (VU.Vector Double))
toRowsVec tensor = do
  -- TODO: use vectors directly instead of going through lists with toRows
  row_values <- toRows tensor
  return $ fmap VU.fromList $ V.fromList row_values

fromRowsVec :: V.Vector (VU.Vector Double) -> IO Tensor
fromRowsVec row_values = do
  let row_values' = fmap VU.toList $ V.toList row_values
  fromRows row_values'

-- copies contents of one tensor to another. Tensor dimensions must match.
-- copy dst src
copy :: Stream -> Tensor -> Tensor -> IO ()
copy stream dst src = do
  when (tensorRows dst /= tensorRows src) $
    error "Destination tensor has incompatible dimensions"
  when (tensorCols dst /= tensorCols src) $
    error "Destination tensor has incompatible dimensions"
  withStreamPtr stream $ \stream_ptr ->
    withTensorPtr dst $ \dst_ptr ->
      withTensorPtr src $ \src_ptr ->
        c_cuda_copy_from_device_to_device_2d
          dst_ptr
          (fromIntegral $ tensorPitch dst)
          src_ptr
          (fromIntegral $ tensorPitch src)
          (fromIntegral $ tensorRows dst)
          (fromIntegral $ tensorCols dst)
          stream_ptr

-- subtract two matrices
subtract :: Stream -> Tensor -> Tensor -> Tensor -> IO ()
subtract stream dst mat1 mat2 = do
  void initCuda
  when (tensorRows mat1 /= tensorRows mat2) $
    error "Cannot subtract matrices with incompatible dimensions"
  when (tensorCols mat1 /= tensorCols mat2) $
    error "Cannot subtract matrices with incompatible dimensions"
  when (tensorRows dst /= tensorRows mat1) $
    error "Destination matrix has incompatible dimensions"
  when (tensorCols dst /= tensorCols mat1) $
    error "Destination matrix has incompatible dimensions"
  withStreamPtr stream $ \stream_ptr ->
    withTensorPtr dst $ \dst_ptr ->
      withTensorPtr mat1 $ \mat1_ptr ->
        withTensorPtr mat2 $ \mat2_ptr ->
          c_cuda_sub dst_ptr (fromIntegral (tensorPitch dst))
                     mat1_ptr (fromIntegral (tensorPitch mat1))
                     mat2_ptr (fromIntegral (tensorPitch mat2))
                     (fromIntegral $ tensorRows dst)
                     (fromIntegral $ tensorCols dst)
                     stream_ptr

-- subtract two matrices
add :: Stream -> Tensor -> Tensor -> Tensor -> IO ()
add stream dst mat1 mat2 = do
  void initCuda
  when (tensorRows mat1 /= tensorRows mat2) $
    error "Cannot add matrices with incompatible dimensions"
  when (tensorCols mat1 /= tensorCols mat2) $
    error "Cannot add matrices with incompatible dimensions"
  when (tensorRows dst /= tensorRows mat1) $
    error "Destination matrix has incompatible dimensions"
  when (tensorCols dst /= tensorCols mat1) $
    error "Destination matrix has incompatible dimensions"
  withStreamPtr stream $ \stream_ptr ->
    withTensorPtr dst $ \dst_ptr ->
      withTensorPtr mat1 $ \mat1_ptr ->
        withTensorPtr mat2 $ \mat2_ptr ->
          c_cuda_add dst_ptr (fromIntegral (tensorPitch dst))
                     mat1_ptr (fromIntegral (tensorPitch mat1))
                     mat2_ptr (fromIntegral (tensorPitch mat2))
                     (fromIntegral $ tensorRows dst)
                     (fromIntegral $ tensorCols dst)
                     stream_ptr

outerProduct :: Stream -> Tensor -> Tensor -> Tensor -> IO ()
outerProduct stream dst vec1 vec2 = do
  cublas_ptr <- initCuda
  when (tensorCols vec1 /= 1) $
    error "Cannot compute outer product of a non-vector"
  when (tensorCols vec2 /= 1) $
    error "Cannot compute outer product of a non-vector"
  when (tensorRows dst /= tensorRows vec1) $
    error "Destination matrix has incompatible dimensions"
  when (tensorCols dst /= tensorRows vec2) $
    error "Destination matrix has incompatible dimensions"

  withBlasLock $
    withStreamPtr stream $ \stream_ptr ->
      withTensorPtr dst $ \dst_ptr ->
        withTensorPtr vec1 $ \vec1_ptr ->
          withTensorPtr vec2 $ \vec2_ptr ->
            c_cuda_outer_product cublas_ptr
                                 dst_ptr (fromIntegral (tensorPitch dst))
                                 vec1_ptr (fromIntegral (tensorPitch vec1))
                                 vec2_ptr (fromIntegral (tensorPitch vec2))
                                 (fromIntegral $ tensorRows dst)
                                 (fromIntegral $ tensorCols dst)
                                 stream_ptr

-- dst mat1 mat2
matMul :: Stream -> Tensor -> Tensor -> Tensor -> IO ()
matMul stream dst mat1 mat2 = do
  cublas_ptr <- initCuda
  when (tensorCols mat1 /= tensorRows mat2) $
    error "Cannot multiply matrices with incompatible dimensions"
  when (tensorRows dst /= tensorRows mat1) $
    error "Destination matrix has incompatible dimensions"
  when (tensorCols dst /= tensorCols mat2) $
    error "Destination matrix has incompatible dimensions"
  withBlasLock $
    withStreamPtr stream $ \stream_ptr ->
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
                          stream_ptr

-- matrix-vector multiply, third tensor is expected to be a column vector
-- dst mat1 vec2.
--
-- result will be given as another column vector
matMulVec :: Stream -> Tensor -> Tensor -> Tensor -> IO ()
matMulVec stream dst mat1 vec2 = do
  cublas_ptr <- initCuda
  when (tensorCols mat1 /= tensorRows vec2) $
    error "Cannot multiply matrices with incompatible dimensions"
  when (tensorRows dst /= tensorRows mat1) $
    error "Destination matrix has incompatible dimensions"
  when (tensorCols dst /= 1) $
    error "Destination matrix has incompatible dimensions (1 column expected)"
  withBlasLock $
    withStreamPtr stream $ \stream_ptr ->
      withTensorPtr dst $ \dst_ptr ->
        withTensorPtr mat1 $ \mat1_ptr ->
          withTensorPtr vec2 $ \vec2_ptr ->
            c_cuda_matmul_vec cublas_ptr
                              dst_ptr (fromIntegral (tensorPitch dst))
                              mat1_ptr (fromIntegral (tensorPitch mat1))
                              vec2_ptr (fromIntegral (tensorPitch vec2))
                              (fromIntegral $ tensorRows dst)
                              (fromIntegral $ tensorCols mat1)
                              stream_ptr

matMulBatched :: Stream -> [Tensor] -> [Tensor] -> [Tensor] -> IO ()
matMulBatched stream dsts mat1s mat2s = matMulBatchedAdd stream dsts mat1s mat2s 0

matMulBatchedAdd :: Stream -> [Tensor] -> [Tensor] -> [Tensor] -> Double -> IO ()
matMulBatchedAdd stream dsts mat1s mat2s multiplier = do
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


  withBlasLock $
    withStreamPtr stream $ \stream_ptr ->
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
                                  stream_ptr

scale :: Stream -> Tensor -> Tensor -> IO ()
scale stream tensor scalar = do
  when (tensorRows scalar /= 1 || tensorCols scalar /= 1) $
    error "Scalar must be a 1x1 tensor"
  void initCuda
  withStreamPtr stream $ \stream_ptr ->
    withTensorPtr tensor $ \tensor_ptr ->
      withTensorPtr scalar $ \scalar_ptr ->
        c_cuda_scale tensor_ptr
                     scalar_ptr
                     (fromIntegral $ tensorPitch tensor)
                     (fromIntegral $ tensorRows tensor)
                     (fromIntegral $ tensorCols tensor)
                     stream_ptr

-- Apply sigmoid function to a tensor
sigmoid :: Stream -> Tensor -> IO ()
sigmoid stream tensor = do
  withStreamPtr stream $ \stream_ptr ->
    withTensorPtr tensor $ \tensor_ptr ->
      c_cuda_sigmoid tensor_ptr
                     (fromIntegral $ tensorPitch tensor)
                     (fromIntegral $ tensorRows tensor)
                     (fromIntegral $ tensorCols tensor)
                     stream_ptr

sigmoidTanh :: Stream -> Tensor -> IO ()
sigmoidTanh stream tensor = do
  withStreamPtr stream $ \stream_ptr ->
    withTensorPtr tensor $ \tensor_ptr ->
      c_cuda_sigmoid_tanh tensor_ptr
                          (fromIntegral $ tensorPitch tensor)
                          (fromIntegral $ tensorRows tensor)
                          (fromIntegral $ tensorCols tensor)
                          stream_ptr
