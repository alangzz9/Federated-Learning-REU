
n = 6000;
d = 10;
r = 5;
cond = 1e5;

UA = orth(randn(d,d));
UB = orth(randn(d,d));
UC = randn(d,d);

D = diag([linspace(1/cond, 1,r) zeros(1,d-r)]);
SigmaA = UA * D * UA';
SigmaB = UB * D * UB';
SigmaC = 0.1 *  UC * UC';
A =  mvnrnd(zeros(d,1), SigmaA,n)';
B =  mvnrnd(zeros(d,1), SigmaB,n)';
C =  mvnrnd(zeros(d,1), SigmaC,n)';

[dim,samples] = size(C)
disp(A(:,1))