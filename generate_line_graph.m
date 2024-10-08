function [Adj, degree, num_of_edge,A,B,D,Lm,edge_index, eig_Lm,min_eig_Lm,WW,LN,L_hat,eig_L_hat,min_eig_L_hat] = generate_line_graph(N,Radius,n )
seed = 1;
[Adj,degree]=linegraph(N,2);
%% incidence matrix
num_of_edge = sum(sum(Adj))/2;
A = zeros(num_of_edge,N);
edge_index = 1;
for ii = 1 : N
    for jj = 1 : N
        if (jj>ii)
            if Adj(ii,jj) == 1
                A(edge_index,jj) = 1;
                A(edge_index,ii) = -1;
                edge_index = edge_index + 1;
            end
        end
    end
end
A = kron(A,eye(n));
B = abs(A);
Lm = A' * A; % signed LAplacian
D = (B' * B + A' * A)/2;


eig_Lm = eig(Lm);
for ii = 1 : length(eig_Lm)
    if (eig_Lm(ii)>=1e-10)
        min_eig_Lm = eig_Lm(ii);
        break;
    end
end

WW = zeros(size(Adj));
for ii = 1:size(Adj,1)
    for jj = 1:size(Adj,1)
        if Adj(ii, jj) == 1
            WW(ii,ii) = WW(ii,ii)+1/sqrt(degree(ii)*degree(jj));
        end
    end
end
WW = kron(WW, eye(n));

LN = D^(-1/2)*(A'*A)*D^(-1/2);
L_hat = LN - diag(diag(LN)) + diag(diag(WW));
eig_L_hat = eig(L_hat);
for ii = 1 : length(eig_L_hat)
    if (eig_L_hat(ii)>=1e-10)
        min_eig_L_hat = eig_L_hat(ii);
        break;
    end
end

end

function [A,degree]=linegraph(n,radius)

xy = [(1:n)', ones(n,1)];
%--> distance matrix of all pairs of nodes
Md=sqrt( (xy(:,1)*ones(1,n)-ones(n,1)*xy(:,1).').^2+(xy(:,2)*ones(1,n)-ones(n,1)*xy(:,2).').^2);
%---> Adjacency matrix
A=((Md+2*radius*eye(n))<radius)*eye(n);

degree=A*ones(n,1);
%---> Laplacian
end