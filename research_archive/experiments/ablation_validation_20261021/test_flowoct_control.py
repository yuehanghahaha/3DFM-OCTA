"""Test actual author update, physical axes and OCT-only generation adapters."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from flowoct_control import to_flowoct,from_flowoct,predict,make_flowoct,read
import torch


class FlowControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_axis_adapter_is_lossless_and_preserves_Bscan_axis(self):
        x=torch.arange(2*1*16*32*48).reshape(2,1,16,32,48)
        native=to_flowoct(x)
        self.assertEqual(native.shape,(2,1,32,48,16))
        torch.testing.assert_close(from_flowoct(native),x,rtol=0,atol=0)
        torch.testing.assert_close(native[0,0,5],x[0,0,:,5,:].T,rtol=0,atol=0)

    def test_two_solver_adapters_use_only_oct(self):
        class Oracle:
            def validation(self,condition,gt,chunk):
                assert gt is None
                return (condition+.1).clamp(0,1)
            def euler_step(self,state,t0,t1):
                return state+(t1-t0)[:,None,None,None]*.1
        x=torch.full((1,1,16,32,48),.2)
        for mode in ('native3','shared8'):
            torch.testing.assert_close(predict(Oracle(),x,mode),x+.1,atol=1e-6,rtol=0)

    def test_real_author_model_update_and_projection_loss_are_finite(self):
        torch.manual_seed(12)
        config=read(ROOT/'06_flowoct/config.json')
        model=make_flowoct(config,torch.device('cpu'))
        x=torch.rand(1,1,32,16,32)
        y=x*.6
        model.set_input(dict(A=to_flowoct(x),B=to_flowoct(y)))
        before={name:p.detach().clone() for name,p in model.netG.named_parameters()}
        model.forward()
        self.assertTrue(torch.isfinite(model.loss))
        self.assertEqual(model.opt.network_chunk,0)
        self.assertEqual(model.opt.train_slices,0)
        values=model.get_current_loss()
        expected=10*values['vloss']+.5*values['L1loss']+.005*values['contrastloss']+.1*values['proj_loss']
        self.assertAlmostEqual(model.loss.item(),expected,places=5)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(any(not torch.equal(before[name],p.detach()) for name,p in model.netG.named_parameters()))
        model.eval()
        for mode in ('native3','shared8'):
            result=predict(model,x,mode,chunk_size=2)
            self.assertEqual(result.shape,x.shape)
            self.assertTrue(torch.isfinite(result).all())


if __name__=='__main__':
    unittest.main(verbosity=2)
