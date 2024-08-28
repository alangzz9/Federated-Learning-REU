function [Opt_grad, Obj, Oracle] = DSGDM_noconsensus(stepsize,PW,  y_temp, iter_num, n, N, lambda, d0, d1, d2, momentum_coeff, features, labels,  bs, minibatch)
Opt_grad = zeros(iter_num-1,1);
Obj = zeros(iter_num-1,1);
Oracle = zeros(iter_num-1,1);

x = reshape(y_temp(:,1),[n, N]);
grad = zeros(n,N);
momentum = zeros(n,N);

alpha = stepsize * sqrt(N/iter_num);

for iter  = 2 : iter_num
    % calculating the opt-gap
    gradient_matrix = zeros(n,N);
    for ii = 1 : N
        sample = 1:bs;
        jj = (ii-1)*bs + sample ;
        gradient_matrix(:,ii) = nnbackward(features(jj,:), labels(jj,:), x(:,ii), d0, d1, d2, lambda, N);
    end
    
    full_grad = mean(gradient_matrix,2);
    Opt_grad(iter-1,1) = norm(full_grad)^2;
    
    %Opt(iter-1,1) = norm(full_grad)^2+1/N *Constraint(iter-1,1);

    
    for ii = 1 : N
        sample = 1:bs;
        jj = (ii-1)*bs + sample ;
        Obj(iter-1,1) = nnforward(features(jj,:), labels(jj,:), x(:,ii), d0, d1, d2, lambda, N);
        
    end
    
    
%     alpha = stepsize;  
    
    sample = randi(bs,1,minibatch);
    Oracle(iter, 1) = Oracle(iter-1, 1) + minibatch;
    
    for ii = 1 : N
        jj = (ii-1)*bs + sample ;
        grad(:, ii) = nnbackward(features(jj,:), labels(jj,:), x(:,ii), d0, d1, d2, lambda, N);
    end
    momentum = momentum_coeff * momentum + grad;
    
    x = x - alpha * momentum;
    x = x * PW;
    momentum = momentum * PW;
%     x = (x * PW -  alpha * momentum ) ;
    
end
end
