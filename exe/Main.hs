module Main where

import Control.Concurrent
import Control.Concurrent.Async
import Control.Monad
import Data.Foldable
import Data.Time
import Data.Traversable
import Prelude hiding ( subtract )
import System.Mem
import System.Random
import Tensor

-- Self-referential meta-learning
--
-- This neural network will output its own weight matrix.
data MetaNN = MetaNN
  { metaWeights :: !Tensor
  , metaConf :: !MetaNNConfiguration }
  deriving ( Show )

data MetaNNConfiguration = MetaNNConfiguration
  { inputSize :: !Int
  , outputSize :: !Int }
  deriving ( Show )

internalOutputsSize :: MetaNNConfiguration -> Int
internalOutputsSize conf = outputSize conf + inputSize conf * 2 + 4

newMetaNN :: MetaNNConfiguration -> IO MetaNN
newMetaNN conf = do
  weights <- zeros (internalOutputsSize conf) (inputSize conf)
  vals <- toRows weights
  randomized_vals <- for vals $ \row -> do
    for row $ \_ -> randomRIO (-1, 1)
  weights <- fromRows randomized_vals
  return $ MetaNN { metaWeights = weights
                  , metaConf = conf }

propagate :: MetaNN -> Tensor -> IO (Tensor, MetaNN)
propagate nn inputs = do
  outputs <- zeros (internalOutputsSize conf) 1

  sigmoidTanh inputs
  matMulVec outputs (metaWeights nn) inputs

  let learning_rate = viewColumnVecOffset outputs 4 0
      query = viewColumnVecOffset outputs (inputSize conf) 4
      key = viewColumnVecOffset outputs (inputSize conf) (4 + inputSize conf)
      output = viewColumnVecOffset outputs (outputSize conf) (4 + inputSize conf * 2)

  sigmoidTanh query
  sigmoidTanh key
  sigmoid learning_rate

  hat_vt <- zeros (internalOutputsSize conf) 1
  vt <- zeros (internalOutputsSize conf) 1
  matMulVec hat_vt (metaWeights nn) key
  matMulVec vt (metaWeights nn) query
  sigmoidTanh hat_vt
  sigmoidTanh vt

  new_weights <- zeros (internalOutputsSize conf) (inputSize conf)
  subtract vt vt hat_vt
  outerProduct new_weights vt key

  subtract new_weights new_weights (metaWeights nn)
  return (output, nn { metaWeights = new_weights })
 where
  conf = metaConf nn

main :: IO ()
main = forever $ do
  meta <- newMetaNN (MetaNNConfiguration 20 1)
  -- random input
  inp <- replicateM 20 $ randomRIO (-1, 1)
  inputs <- fromRows (fmap (:[]) inp)
  (result, _) <- propagate meta inputs
  r <- toRows result
  print r
