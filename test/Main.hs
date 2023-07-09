module Main ( main ) where

import Data.Foldable
import Data.Traversable
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VU
import System.IO.Unsafe
import System.Random
import Test.Hspec
import Test.QuickCheck

import Tensor

main :: IO ()
main = hspec $ do
  tensorTests

data ViewTestCase = ViewTestCase !Tensor !Int !Int !Int !Int
  deriving ( Show )

instance Arbitrary ViewTestCase where
  arbitrary = do
    t <- arbitrary
    x <- choose (0, rows t - 1)
    y <- choose (0, cols t - 1)
    nrows <- choose (1, rows t - x)
    ncols <- choose (1, cols t - y)
    return $ ViewTestCase t x y nrows ncols

data MatMulTestCase = MatMulTestCase !Tensor !Tensor
  deriving ( Show )

instance Arbitrary MatMulTestCase where
  {-# NOINLINE arbitrary #-}
  arbitrary = do
    t1_rows <- choose (1, 200)
    common_dim <- choose (1, 200)
    t2_cols <- choose (1, 200)

    return $ unsafePerformIO $ do
      t1 <- zeros nullStream t1_rows common_dim
      t2 <- zeros nullStream common_dim t2_cols

      t1_rows <- toRowsVec t1
      t2_rows <- toRowsVec t2

      t1_rows_randomized <- for t1_rows $ \row ->
        VU.fromList <$> for (VU.toList row) (\_ -> randomRIO (-2.0, 2.0))
      t2_rows_randomized <- for t2_rows $ \row ->
        VU.fromList <$> for (VU.toList row) (\_ -> randomRIO (-2.0, 2.0))

      t1_final <- fromRowsVec t1_rows_randomized
      t2_final <- fromRowsVec t2_rows_randomized

      return $ MatMulTestCase t1_final t2_final

tensorTests :: Spec
tensorTests = do
  describe "Tensor" $ do
    it "Allocating tensors works" $
      property $ \nrows ncols -> nrows > 0 && ncols > 0 && nrows < 200 && ncols < 200 ==> do
        s <- makeStream
        t <- zeros s nrows ncols
        rows t `shouldBe` nrows
        cols t `shouldBe` ncols

    it "toRows and fromRows are identity" $
      property $ \tensor -> do
        row_values <- toRows tensor
        tensor2 <- fromRows row_values
        rows tensor `shouldBe` rows tensor2
        cols tensor `shouldBe` cols tensor2
        row_values2 <- toRows tensor2
        row_values `shouldBe` row_values2

    it "view works correctly" $
      property $ \(ViewTestCase tensor x y nrows ncols) ->
        (x + nrows <= rows tensor &&
         y + ncols <= cols tensor &&
         x >= 0 && y >= 0 && nrows > 0 && ncols > 0) ==> do
          tensor_values <- toRowsVec tensor
          let viewed = view tensor x y nrows ncols
          viewed_values <- toRowsVec viewed

          for_ [0..nrows-1] $ \viewed_x ->
            for_ [0..ncols-1] $ \viewed_y -> do
              let original_x = viewed_x + x
                  original_y = viewed_y + y
              (tensor_values V.! original_x VU.! original_y) `shouldBe` (viewed_values V.! viewed_x VU.! viewed_y)

    it "matMul works" $
      property $ \(MatMulTestCase t1 t2) -> do
        t3 <- zeros nullStream (rows t1) (cols t2)
        matMul nullStream t3 t1 t2

        t1_rows <- toRowsVec t1
        t2_rows <- toRowsVec t2

        let t3_rows_manually = V.generate (rows t1) $ \i ->
              VU.generate (cols t2) $ \j ->
                VU.sum $ VU.zipWith (*) (t1_rows V.! i) (toUnboxed $ fmap (VU.! j) t2_rows)

        t3_rows <- toRowsVec t3

        -- the results may not be 100% equal because the Tensor uses float16
        -- but pure Haskell code t3_rows_manually above is using 64-bit doubles
        -- in the computation 
        --
        -- Especially for larger matrices the error can be quite large.
        for_ [0..rows t1-1] $ \i ->
          for_ [0..cols t2-1] $ \j ->
            (t3_rows V.! i VU.! j) `shouldSatisfy` (\x -> abs (x - (t3_rows_manually V.! i VU.! j)) < 0.2)

    it "gaussian works" $
      property $ \tensor -> do
        rand_tensor <- gaussian nullStream 123 (rows tensor) (cols tensor) 0 0.1
        rand_tensor_vec <- toRowsVec rand_tensor

        for_ rand_tensor_vec $ \vec ->
          for_ (VU.toList vec) $ \value ->
            value `shouldSatisfy` (\x -> x >= -1.0 && x <= 1.0)

        rand_tensor <- gaussian nullStream 123 (rows tensor) (cols tensor) 3.0 0.1
        rand_tensor_vec <- toRowsVec rand_tensor

        for_ rand_tensor_vec $ \vec ->
          for_ (VU.toList vec) $ \value ->
            value `shouldSatisfy` (\x -> x >= 2.0 && x <= 4.0)


toUnboxed :: VU.Unbox a => V.Vector a -> VU.Vector a
toUnboxed = VU.fromList . V.toList
