function [Oracle_outer, return_AUC_outer, return_AUC1_outer,Oracle_inner, return_AUC_inner, return_AUC1_inner] = ...
     algorithm_3 ( stepsize_x, stepsize_y, PW, x_temp, alpha_temp, iter_num, n,nodes_num, ...
    gc_v_1,gc_v_0,gc_alpha_1,gc_alpha_0, features, labels, bs, minibatch,x_test,y_test,rho,R)


iter_num = 1000;
Oracle_outer = zeros(iter_num+1 ,1);
return_AUC_outer = zeros(iter_num+1,1);
return_AUC1_outer = zeros(iter_num+1,1);
Oracle_outer(1,1) = 0;

Oracle_inner = zeros(iter_num*R+1 ,1);
return_AUC_inner = zeros(iter_num*R+1,1);
return_AUC1_inner = zeros(iter_num*R+1,1);
Oracle_inner(1,1) = 0;

N=nodes_num;

nT=ones(1,N);%
p0=zeros(1,N);%
p=zeros(1,N);
gamma = 0.1;
lambda = 0.001;

% g is for primal variable, h is for dual variable
v = zeros(n+2,N);
u = zeros(1,N);

g = zeros(n+2,bs,N);
h = zeros(1,bs,N);

aa = zeros(n+2,N);
bb= zeros(1,N);


eta = stepsize_x;
eta_y = stepsize_y;    

x_list = zeros(n+2, N, R);
y_list = zeros(1, N, R);
for iter = 1:iter_num % outer loop start
    if iter == 1

        x = reshape(x_temp(:,1),[n, N]);% x_temp and n is parameter number?
        a = randn(1,N);
        b = randn(1,N);
        
        % 'x_large' primal variables
        x_large = [x;a;b];
        x_large_old = x_large;
        % 'y' dual variables
        y = reshape(alpha_temp(:,1),[1, N]);%alpha_temp?
        y_old = y;
    else
        % compute the x_mean and y mean
%         r_selected = randi(R,1);
        r_selected = R;
        x_temp = mean(x_list(:, :, r_selected), 2);


        %x_temp = reshape(x_temp, n+2);
        x_large = zeros(n+2, N);
        for ii = 1:N
            x_large(:, ii) = x_temp;
        end
        
        y_temp = mean(y_list(:, :, r_selected), 2);
        
        %y_temp = reshape(y_temp, 1);
        y = zeros(1, N);
        for ii = 1:N
            y(:, ii) = y_temp;
        end
    end

    if iter == 1 
        % use the x_mean to compute the auc
        x_cal = mean(x_large(1:n,:),2);% the mean weight of network
        y_pred_train=features'*x_cal;
        y_pred = x_test'*x_cal;
        [X,Y,T,AUC1] = perfcurve(labels,y_pred_train,1);
        [X,Y,T,AUC] = perfcurve(y_test,y_pred,1);
        
        return_AUC1_outer(1,1)=AUC1;
        return_AUC_outer(1,1)=AUC;
        return_AUC1_inner(1,1)=AUC1;
        return_AUC_inner(1,1)=AUC;
    end

    % inner loop start
        
    % 'sample0' index of selected data point
    sample0 = randi(bs,1,minibatch);
    % get the mean of all samples in bs
    g_mean = mean(g,2);
    g_mean = reshape(g_mean,[n+2,N]);
    h_mean = mean(h,2);
    h_mean = reshape(h_mean,[1,N]);
    
    % get mean of selected samples
    g_sample_mean = mean(g(:,sample0,:),2);
    g_sample_mean = reshape(g_sample_mean,[n+2,N]);
    h_sample_mean = mean(h(:,sample0,:),2);
    h_sample_mean = reshape(h_sample_mean,[1,N]);
    
    
    grad_v_temp = zeros(n+2,N);
    grad_v_old_temp = zeros(n+2,N);
    grad_alpha_temp = zeros(1,N);
    grad_alpha_old_temp = zeros(1,N);
    
    grad_v = zeros(n+2,N);
    grad_v_old = zeros(n+2,N);
    
    grad_alpha = zeros(1,N);
    grad_alpha_old = zeros(1,N);

    Oracle_outer(iter+1, 1) = Oracle_outer(iter+1, 1) + length(sample0);
    for ii = 1 : N 
        % index of random minibatch
        sample= sample0 + (ii-1)*bs;
        for i=1:size(sample,2)% go througn the mnbatch
            jj = sample(i);
    
            if labels(sample(i))==1
        
                p(1,ii) = (p0(1,ii)*(nT(1,ii)-1)+1)/nT(1,ii);

                gcx_1 = gc_v_1(x_large(:,ii),features(:,jj),[-1;0],[0;0], y(:,ii),p(1,ii));
                gcy_1 = gc_alpha_1(x_large(:,ii),features(:,jj),[0;0], y(:,ii),p(1,ii));

                gcx_1_old = gc_v_1(x_large_old(:,ii),features(:,jj),[-1;0],[0;0], y_old(:,ii),p(1,ii));
                gcy_1_old = gc_alpha_1(x_large_old(:,ii),features(:,jj),[0;0], y_old(:,ii),p(1,ii));

                grad_v_temp(:,ii) = grad_v_temp(:,ii) + gcx_1;
                grad_v_old_temp(:,ii) = grad_v_old_temp(:,ii) + gcx_1_old;

                grad_alpha_temp(:,ii) = grad_alpha_temp(:,ii) +gcy_1;
                grad_alpha_old_temp(:,ii) = grad_alpha_old_temp(:,ii) + gcy_1_old;

                g(:,sample0(i),ii) = gcx_1;
                h(:,sample0(i),ii) = gcy_1;
   
    
            else
        
                p(1,ii) = (p0(1,ii)*(nT(1,ii)-1))/nT(1,ii);
                gcx_0 = gc_v_0(x_large(:,ii),features(:,jj),[0;-1],[0;0], y(:,ii),p(1,ii));
                gcy_0 = gc_alpha_0(x_large(:,ii),features(:,jj),[0;0], y(:,ii),p(1,ii));

                gcx_0_old = gc_v_0(x_large_old(:,ii),features(:,jj),[0;-1],[0;0], y_old(:,ii),p(1,ii));
                gcy_0_old = gc_alpha_0(x_large_old(:,ii),features(:,jj),[0;0], y_old(:,ii),p(1,ii));
                
                grad_v_temp(:,ii) = grad_v_temp(:,ii)+gcx_0;
                grad_v_old_temp(:,ii) = grad_v_old_temp(:,ii)+gcx_0_old;
                grad_alpha_temp(:,ii) = grad_alpha_temp(:,ii)+gcy_0;
                grad_alpha_old_temp(:,ii) = grad_alpha_old_temp(:,ii) + gcy_0_old;
                
                g(:,sample0(i),ii) = gcx_0;
                h(:,sample0(i),ii) = gcy_0;
            end
            nT(1,ii)=nT(1,ii)+1;
            p0(1,ii)=p(1,ii);
        end 
            grad_v(:,ii) = grad_v_temp(:,ii)/length(sample0);
            grad_alpha(:,ii) = grad_alpha_temp(:,ii)/length(sample0);
            grad_v_old(:,ii) = grad_v_old_temp(:,ii)/length(sample0);
            grad_alpha_old(:,ii) = grad_alpha_old_temp(:,ii)/length(sample0);
    end
    v_old = v;
    u_old = u;
    % compute the v and u
    v =  (1-rho)*(v_old + grad_v - grad_v_old)+ rho* (grad_v_old - g_sample_mean + g_mean);
    u =  (1-rho)* (u_old + grad_alpha -grad_alpha_old) + rho* (grad_alpha_old - h_sample_mean + h_mean);
    aa =  aa* PW + v - v_old;
    bb =  bb * PW + u - u_old;

    x_large_old = x_large;
    x_temp = x_large - gamma*aa;
    a = lambda*gamma;
    result = zeros(size(x_temp));
    result(x_temp > a) = x_temp(x_temp > a) - a;
    result(x_temp < -a) = x_temp(x_temp < -a) + a;
    result = result*PW;
    x_large = x_large + eta * (result - x_large);


    y_old = y;
    y_temp = y*PW + gamma* bb;
    y = y + eta_y * (y_temp - y);

    % compute the global auc
    x_cal = mean(x_large(1:n,:),2);% the mean weight of network
    y_pred_train=features'*x_cal;
    y_pred = x_test'*x_cal;
    [X,Y,T,AUC1] = perfcurve(labels,y_pred_train,1);
    [X,Y,T,AUC] = perfcurve(y_test,y_pred,1);

    return_AUC1_outer(iter+1,1)=AUC1;
    return_AUC_outer(iter+1,1)=AUC;
end
disp(iter)



end
%disp(return_AUC_outer)
%disp(return_accuracy)
%disp(Oracle)


