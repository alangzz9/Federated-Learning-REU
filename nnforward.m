function [loss, h1, output] = nnforward(X, Y, W, d0, d1, d2, lambda, num_devices)


W1 = reshape(W(1:d0*d1), [d0, d1]);
W2 = reshape(W(d0*d1+1:end), [d1, d2]);


z1 = X * W1;  %x: nxd0; w1: d0xd1
h1 = sigmoid(z1); %nxd1

% h1 = [h1 ones(size(X,1),1)]; %nx(d1+1) 

z2 = h1 * W2; %nxd2; W2: d1xd2
output = softmax(z2);

prediction = sum(output .* Y, 2);

% [~, index] = find(Y);
% predict = output(sub2ind(size(output), 1:size(output, 1), index));

% loss = -sum(log(prediction)) + lambda * sum(W.*W);
loss = -sum(log(prediction)) +  lambda * sum(W.^2 ./ (1+W.^2));
loss = loss/(size(X, 1)*num_devices);
end

function y = sigmoid(x)

    y = 1.0 ./ (1.0 + exp(-x));
end
