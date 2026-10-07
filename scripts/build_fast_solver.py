"""Regenerate the compiled numerical kernel from the reference STATSTAR source.

Keep equations, shell limits, step rules and acceptance tests identical. Remove
printing/text output and replace the constant container with literal constants.
"""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=ast.parse((root/'starmodel.py').read_text())
constants=dict(sigma=5.67051e-5,c=2.99792458e10,a=7.56591e-15,G=6.67259e-8,
               k_B=1.380658e-16,m_H=1.673534e-24,gamma=5/3,g_ff=1.,
               Rsun=6.9599e10,Msun=1.989e33,Lsun=3.826e33,gamrat=(5/3)/((5/3)-1),kPad=.3)
class NumericalOnly(ast.NodeTransformer):
    def visit_If(self,node):
        if isinstance(node.test,ast.Name) and node.test.id=='verbose':return None
        if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='debug':return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='print':return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='cst' for t in node.targets):return None
        return self.generic_visit(node)
    def visit_Attribute(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='cst':return ast.copy_location(ast.Constant(constants[node.attr]),node)
        return self.generic_visit(node)
funcs=[]
for fn in source.body:
    if not isinstance(fn,ast.FunctionDef) or fn.name=='main':continue
    if fn.name=='starmodel':
        fn.name='integrate'
        # Retain through the four unconditional core extrapolation statements.
        cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rhocor' for t in n.targets) and i>40)
        fn.body=ast.parse('istop=0; ip1=0').body+fn.body[:cut+4]+ast.parse('return Igoof,ierr,istop,r,P,M_r,L_r,T,rho,kappa,epslon,dlPdlT,rhocor,epscor,Pcore,Tcore').body
        fn.args=ast.parse('def f(Msolar,Lsolar,Te,X,Z): pass').body[0].args
    fn=NumericalOnly().visit(fn)
    fn.decorator_list=ast.parse('@njit(cache=True, error_model="numpy")\ndef f(): pass').body[0].decorator_list
    funcs.append(fn)
module=ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[]))
header='"""Generated numerical kernel; regenerate with scripts/build_fast_solver.py.\nDerived from starmodel.py (STATSTAR; Schiminovich/Nidever). No fastmath.\n"""\nimport numpy as np\nfrom numba import njit\n\n'
(root/'fast_kernel.py').write_text(header+ast.unparse(module)+'\n')
