module LSTM
  ( NumInputs
  , NumOutputs
  , LayerSize
  , LSTM()
  , LSTMState()
  , newLSTMState
  , newLSTM
  , propagate )
  where

import Control.Monad
import Data.Traversable
import Data.Vector ( Vector, (!) )
import qualified Data.Vector as V
import Tensor

-- ***** TODO: implementation is unverified.
-- Need to write pure Haskell implementation and check are the numbers the
-- same as tests.

type NumInputs = Int
type NumOutputs = Int
type LayerSize = Int

data LSTM = LSTM
  { toOutputLayer :: !Tensor
  , layerConnects :: !(Vector LSTMConnect)
  , hiddenToHiddenConnects :: !(Vector LSTMConnect)
  , biases :: !(Vector LSTMConnect)
  , layerSizes :: !(Vector LayerSize)
  , widestLayer :: !Int
  , numInputs :: !Int
  , numOutputs :: !Int }
  deriving ( Show )

data LSTMConnect = LSTMConnect
  { inputsToForgetGate :: !Tensor
  , inputsToInputGate :: !Tensor
  , inputsToOutputGate :: !Tensor
  , inputsToCell :: !Tensor }
  deriving ( Show )

data LSTMState = LSTMState
  { targetsInputGate :: !Tensor
  , targetsOutputGate :: !Tensor
  , targetsForgetGate :: !Tensor
  , targetsCell :: !Tensor
  , targetsOutput :: !Tensor
  , finalOutput :: !Tensor
  , memories :: !(Vector Tensor)
  , lastActivations :: !(Vector Tensor)
  , tmpMemories :: !Tensor
  , networkLstm :: !LSTM }
  deriving ( Show )

newLSTMState :: LSTM -> IO LSTMState
newLSTMState lstm = do
  targets_input_gate <- zeros (widestLayer lstm) 1
  targets_output_gate <- zeros (widestLayer lstm) 1
  targets_forget_gate <- zeros (widestLayer lstm) 1
  targets_cell <- zeros (widestLayer lstm) 1
  targets_output <- zeros (max (widestLayer lstm) (numOutputs lstm)) 1
  memories <- for (layerSizes lstm) $ \sz -> zeros sz 1
  last_activations <- for (layerSizes lstm) $ \sz -> zeros sz 1
  tmp_memories <- zeros (widestLayer lstm) 1
  final_output <- zeros (numOutputs lstm) 1
  return LSTMState {
    targetsInputGate = targets_input_gate,
    targetsOutputGate = targets_output_gate,
    targetsForgetGate = targets_forget_gate,
    targetsCell = targets_cell,
    targetsOutput = targets_output,
    finalOutput = final_output,
    memories = memories,
    lastActivations = last_activations,
    tmpMemories = tmp_memories,
    networkLstm = lstm }

makeLSTMConnect :: NumOutputs -> NumInputs -> IO LSTMConnect
makeLSTMConnect noutputs ninputs = do
  fg <- zeros noutputs ninputs
  ig <- zeros noutputs ninputs
  og <- zeros noutputs ninputs
  c <- zeros noutputs ninputs
  return LSTMConnect {
    inputsToForgetGate = fg,
    inputsToInputGate = ig,
    inputsToOutputGate = og,
    inputsToCell = c }

makeLSTMConnect11 :: NumInputs -> IO LSTMConnect
makeLSTMConnect11 ninputs = do
  fg <- zeros ninputs 1
  ig <- zeros ninputs 1
  og <- zeros ninputs 1
  c <- zeros ninputs 1
  return LSTMConnect {
    inputsToForgetGate = fg,
    inputsToInputGate = ig,
    inputsToOutputGate = og,
    inputsToCell = c }

-- | LSTM based on the Tensor module.
--
-- It has one linearly connected input->first LSTM layer and last LSTM layer->output.
--
-- All matrices are initialized with an orthonormal matrix.
--
-- LSTM layers have random values between -1.0 and 1.0.
newLSTM :: NumInputs -> [LayerSize] -> NumOutputs -> IO LSTM
newLSTM ninputs layer_sizes noutputs = do
  when (length layer_sizes == 0) $
    error "LSTM must have at least one layer"

  to_output_layer <- zeros noutputs (last layer_sizes)

  layer_connects <- go (ninputs:layer_sizes)
  hidden_layer_connects <- for layer_sizes makeLSTMConnect11
  biases <- for layer_sizes makeLSTMConnect11

  return LSTM { toOutputLayer = to_output_layer
              , layerSizes = V.fromList layer_sizes
              , layerConnects = V.fromList layer_connects
              , numInputs = ninputs
              , numOutputs = noutputs
              , hiddenToHiddenConnects = V.fromList hidden_layer_connects
              , biases = V.fromList biases
              , widestLayer = maximum layer_sizes }
 where
  go (x:y:rest) = do
    connects <- makeLSTMConnect y x
    rest <- go (y:rest)
    return (connects:rest)
  go _ = return []

propagate :: Tensor -> LSTMState -> IO Tensor
propagate input_tensor st = go input_tensor 0
 where
  lstm = networkLstm st

  go inputs idx | idx >= V.length (layerSizes lstm) = do
    matMul (finalOutput st) (toOutputLayer lstm) inputs
    return $ finalOutput st

  go inputs idx | idx < V.length (layerSizes lstm) = do
    let c = layerConnects lstm ! idx
        noutputs = rows (inputsToCell c)
        bias = biases lstm ! idx
        mems = memories st ! idx
        last_acts = lastActivations st ! idx
        tmp_mems = tmpMemories st
        hidden_wgts = hiddenToHiddenConnects lstm ! idx

    let t_ig = viewColumnVec (targetsInputGate st) noutputs
        t_og = viewColumnVec (targetsOutputGate st) noutputs
        t_o = viewColumnVec (targetsOutput st) noutputs
        t_c = viewColumnVec (targetsCell st) noutputs
        t_fg = viewColumnVec (targetsForgetGate st) noutputs
        t_tmp_mems = viewColumnVec tmp_mems noutputs

    -- Note: there is no connectivity between the LSTMs in the layer, should
    -- add it here to make a vanilla LSTM.
    --
    -- That is, the LSTM nodes in the same layer are not connected to each other. Oops.
    lstmBiasLastAct t_ig (inputsToInputGate bias) (inputsToInputGate hidden_wgts) last_acts
    lstmBiasLastAct t_og (inputsToOutputGate bias) (inputsToOutputGate hidden_wgts) last_acts
    lstmBiasLastAct t_c (inputsToCell bias) (inputsToCell hidden_wgts) last_acts
    lstmBiasLastAct t_fg (inputsToForgetGate bias) (inputsToForgetGate hidden_wgts) last_acts

    -- note: inputs and t_o might be the same tensor, this is fine but be
    -- careful if you make changes.

    matMulBatchedAdd [t_ig, t_og, t_c, t_fg]
                     [inputsToInputGate c
                     ,inputsToOutputGate c
                     ,inputsToCell c
                     ,inputsToForgetGate c]
                     [inputs, inputs, inputs, inputs]
                     1.0

    sigmoid t_ig
    sigmoid t_fg
    sigmoid t_og
    sigmoidTanh t_c

    lstmMemory t_tmp_mems mems t_fg t_c t_ig
    lstmOutput t_o t_tmp_mems t_og

    copy mems t_tmp_mems
    copy last_acts t_o

    go t_o (idx+1)

  go _ _ = error "impossible"
