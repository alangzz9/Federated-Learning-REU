function [Opt_grad, Obj, Oracle] = HSGD_noconsensus(stepsize, beta, PW, y_temp, iter_num, n, N, lambda, d0, d1, d2,  features, labels,  bs, minibatch)
Opt_grad = zeros(iter_num-1,1);
Obj = zeros(iter_num-1,1);
Oracle = zeros(iter_num-1,1);
x = reshape(y_temp(:,1),[n, N]);
grad = zeros(n,N);

grad_old = zeros(n,N);
% gradient = zeros(N*n,1);
% for ii = 1 : N
%     for jj=(ii-1)*bs+1:ii*bs
%         gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
%     end
%     grad(:,ii) = gradient((ii-1)*n+1:ii*n);
% end

sample = randi(bs,1,minibatch);
for ii = 1 : N
    jj = (ii-1)*bs + sample ;
    grad(:, ii) = nnbackward(features(jj,:), labels(jj,:), x(:,ii), d0, d1, d2, lambda, N);
end

v_DGET =  grad;
y_DGET  = grad;

for iter  = 2 : iter_num
    for ii = 1 : N
        sample = 1:bs;
        jj = (ii-1)*bs + sample ;
        Obj(iter-1,1) = nnforward(features(jj,:), labels(jj,:), x(:,ii), d0, d1, d2, lambda, N);

    end
    % calculating the opt-gap
    gradient_matrix = zeros(n,N);
    for ii = 1 : N
        sample = 1:bs;
        jj = (ii-1)*bs + sample ;
        gradient_matrix(:,ii) = nnbackward(features(jj,:), labels(jj,:), x(:,ii), d0, d1, d2, lambda, N);
    end
    
    full_grad = mean(gradient_matrix,2);    
    Opt_grad(iter-1,1) = norm(full_grad)^2;
       
    
    % start
    alpha = stepsize;
%     beta = c * stepsize * stepsize;
    
    % Update x
    x_old = x;
    x = x*PW - alpha * y_DGET;
    
    % Update v
    v_DGET_old = v_DGET;

    sample = randi(bs,1,minibatch);
    Oracle(iter, 1) = Oracle(iter-1, 1) + minibatch;

    for ii = 1 : N
        jj = (ii-1)*bs + sample ;
        grad(:, ii) = nnbackward(features(jj,:), labels(jj,:), x(:,ii), d0, d1, d2, lambda, N);
    end

    for ii = 1 : N
        jj = (ii-1)*bs + sample ;
        grad_old(:, ii) = nnbackward(features(jj,:), labels(jj,:), x_old(:,ii), d0, d1, d2, lambda, N);
    end
    
    v_DGET = (1-beta) * (v_DGET - grad_old) + grad ;
        
    % Update y
    y_DGET = y_DGET*PW + v_DGET - v_DGET_old;
    
    
    
end
end
