function grad = nnbackward(X, Y, W, d0, d1, d2, lambda, num_devices)

W1 = reshape(W(1:d0*d1), [d0, d1]);
W2 = reshape(W(d0*d1+1:end), [d1, d2]);

[~, h1, output] = nnforward(X, Y, W, d0, d1, d2, lambda, num_devices);

dz2 = Y - output; %nxd2
grad2 = h1' * dz2; % d1xd2;

da1 = dz2 * W2'; %nxd1
dz1 = da1 .* h1 .* (1-h1); %nxd1
grad1 = X' * dz1; %d0xd1

grad1 = reshape(grad1, [d0*d1, 1]);
grad2 = reshape(grad2, [d1*d2, 1]);

% grad = [grad1;grad2] + 2 * lambda * W;
grad = [grad1;grad2] + 2 *lambda*W ./ (1+W.^2).^2;

grad = grad./(size(X, 1)*num_devices);

end