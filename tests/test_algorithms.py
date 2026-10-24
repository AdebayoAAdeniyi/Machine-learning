import unittest
import numpy as np
from ml_portfolio.core import ordinary_least_squares, softmax, Standardizer, add_intercept
from ml_portfolio.optimization import fit_logistic,fit_linear_sgd,fit_softmax_sgd
from ml_portfolio.retention import metrics

class TestMethods(unittest.TestCase):
    def test_ols(self):
        x=np.arange(10.)[:,None];y=2+3*x[:,0]
        np.testing.assert_allclose(ordinary_least_squares(x,y),[2,3],atol=1e-9)
    def test_softmax_stability(self):
        self.assertAlmostEqual(float(softmax([[1000,1001]])[0].sum()),1.)
    def test_standardizer_constant(self):
        z=Standardizer().fit([[1,2],[1,4]]).transform([[1,6]])
        self.assertTrue(np.isfinite(z).all())
    def test_logistic(self):
        x=np.array([[-3],[-2],[-1],[1],[2],[3.]]);y=np.array([0,0,0,1,1,1])
        w=fit_logistic(x,y,epochs=300);self.assertTrue(np.array_equal((add_intercept(x)@w>0).astype(int),y))
    def test_linear_sgd(self):
        x=np.arange(-5.,5.)[:,None];y=1+.5*x[:,0]
        w,h=fit_linear_sgd(x,y,x,y,learning_rate=.02,epochs=500)
        self.assertLess(h.best_score,.02)
    def test_softmax(self):
        x=np.array([[-3],[-2],[-1],[1],[2],[3.]]);y=np.array([0,0,0,1,1,1])
        w,h=fit_softmax_sgd(x,y,x,y,epochs=70)
        self.assertGreater(h.best_score,.95)
    def test_metrics(self):
        self.assertAlmostEqual(metrics([0,1,1],[0,1,0])['recall'],.5)
if __name__=='__main__':unittest.main()
