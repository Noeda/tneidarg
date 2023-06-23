module Main where

import Control.Concurrent
import Control.Concurrent.Async
import Data.Foldable
import Data.Time
import Data.Traversable
import LSTM
import System.Mem
import Tensor

main :: IO ()
main = do
  start <- getCurrentTime
  lstm <- newLSTM 40 (replicate 20 200) 3
  st <- newLSTMState lstm
  inp <- zeros 40 1
  for_ [0..4000] $ \idx -> do
      out <- propagate inp st
      out_r <- toRows out
      return ()
  end <- getCurrentTime
  putStrLn $ "Time: " ++ show (diffUTCTime end start)
