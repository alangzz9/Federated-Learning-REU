function [Opt_grad, Opt_concensus, Obj, Oracle] = PSGD(stepsize,PW,  y_temp, iter_num, A, n, N, gc,fc, lambda, aalpha, features, labels,  bs, minibatch)
% n: feature dimension
% N: number of workers
% bs: number of samples on each worker

Opt_grad = zeros(iter_num-1,1);
Opt_concensus = zeros(iter_num-1,1);
Obj = zeros(iter_num-1,1);
Oracle = zeros(iter_num-1, 1);
Constraint = zeros(iter_num-1,1);
x = reshape(y_temp(:,1),[n, N]);
grad = zeros(n,N);

alpha = stepsize * sqrt(N/iter_num);


for iter  = 2 : iter_num
    % calculating the opt-gap
    gradient = zeros(N*n,1);
    gradient_matrix = zeros(n,N);
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        gradient_matrix(:,ii) = gradient((ii-1)*n+1:ii*n);
    end
    
    full_grad = mean(gradient_matrix,2);
    x_vec = reshape(x,[N*n,1]);
    Constraint(iter-1,1) =  norm(A*x_vec(:,1))^2;
    
    Opt_grad(iter-1,1) = norm(full_grad)^2;
    Opt_concensus(iter-1,1) = 1/N *Constraint(iter-1,1);
    
    %Opt(iter-1,1) = norm(full_grad)^2+1/N *Constraint(iter-1,1);

    
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            Obj(iter-1,1) = Obj(iter-1,1) + fc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
    end
    
    
%     alpha = stepsize;  
    gradient = zeros(N*n,1);
    sample = randi(bs,1,minibatch);
    Oracle(iter, 1) = Oracle(iter-1, 1) + minibatch;

    for ii = 1 : N
        for jj=(ii-1)*bs + sample
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        grad(:,ii) = gradient((ii-1)*n+1:ii*n);
    end
      
    x = x * PW -  alpha * grad ;
end
end
