function [Opt_grad, Opt_concensus, Obj, Oracle] = DSGTM(stepsize,PW,  y_temp, iter_num, A, n, N, gc,fc, lambda, aalpha, momentum_coeff, features, labels,  bs, minibatch)
% 'n' problem dimention
% 'N'  nodes_num
% 'bs' num of data points on each worker
Opt_grad = zeros(iter_num-1,1);
Opt_concensus = zeros(iter_num-1,1);
Obj = zeros(iter_num-1,1);
Oracle = zeros(iter_num-1,1);

Constraint = zeros(iter_num-1,1);
x = reshape(y_temp(:,1),[n, N]);x
grad = zeros(n,N);
momentum = zeros(n,N);

alpha = stepsize * sqrt(N/iter_num);

for iter  = 2 : iter_num
    % calculating the opt-gap
    gradient = zeros(N*n,1);
    gradient_matrix = zeros(n,N);
    for ii = 1 : N
        % 'jj' index of all data points on this worker
        for jj=(ii-1)*bs+1:ii*bs 
            % compute the full gradient of this worker
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        gradient_matrix(:,ii) = gradient((ii-1)*n+1:ii*n);
    end
    % compute global gradient of all data on all workers
    full_grad = mean(gradient_matrix,2);
    
    % compute metric
    x_vec = reshape(x,[N*n,1]);
    Constraint(iter-1,1) =  norm(A*x_vec(:,1))^2;
    Opt_grad(iter-1,1) = norm(full_grad)^2;
    Opt_concensus(iter-1,1) = 1/N *Constraint(iter-1,1);
    
    %Opt(iter-1,1) = norm(full_grad)^2+1/N *Constraint(iter-1,1);
    

    % compute the loss function for all workers
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            Obj(iter-1,1) = Obj(iter-1,1) + fc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
    end
    
    
%     alpha = stepsize;  
    
    gradient = zeros(N*n,1);
    % sample is selected index from 1 to bs, length is minibatch
    sample = randi(bs,1,minibatch);
    Oracle(iter, 1) = Oracle(iter-1, 1) + minibatch;

    for ii = 1 : N
        % sample(index) + int (point-wise plus), jj is the index for randomly selected data on each worker
        for jj=(ii-1)*bs + sample
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        % grad size: (n,N)
        grad(:,ii) = gradient((ii-1)*n+1:ii*n);
    end
    
    if iter == 2
        y = grad;
    else
        y = y * PW + grad - grad_old;
    end
    grad_old = grad;

    momentum = momentum_coeff * momentum + y;

    x = x * PW - alpha * momentum;
    
end
end
