function [Adj, A, weights] = generate_graph_fea(num_nodes, pl, num_features)

    Adj = gen_network_topology(num_nodes,pl);  %%% random graph with probability pl and size n

 
    num_of_edge = sum(sum(Adj))/2;
    A = zeros(num_of_edge, num_nodes);
    edge_index = 1;
    
    for ii = 1 : num_nodes
        for jj = 1 : num_nodes
            if (jj>ii)
                if Adj(ii,jj) == 1
                    A(edge_index,jj) = 1;
                    A(edge_index,ii) = -1;
                    edge_index = edge_index + 1;
                end
            end
        end
    end
    edges = kron(A,eye(num_features));
 
    degrees = sum(Adj, 2);
    weights = zeros(num_nodes);
    for i = 1 : num_nodes
        for j = 1 : num_nodes
            if (Adj(i, j) == 0)
                continue;
            end
            weights(i, j)  = 1.0 / (1.0 + max(degrees(i), degrees(j)));
        end
    end
    sum_weights = sum(weights, 2);
    weights = weights + diag(1 - sum_weights);
    
    ROOT = './';
    filename = [ROOT, 'graph_', num2str(num_nodes), '.mat'];
    save(filename, 'weights', 'edges', 'Adj', '-v7.3');
%     save(filename, 'weights', 'Adj', '-v7.3');
    
end

function cmat=gen_network_topology(N,pl)

    % connectivity matrix
    while(true)
        tempmat=rand(N);
        cmat=zeros(N);
        tempmat=(tempmat+tempmat').*(1-eye(N));
        cmat(tempmat>2*(1-pl))=1;
        cmat(tempmat<=2*(1-pl))=0;

        % connectivity
        nodeclass.conmatrix=cmat;
        flag1con=verify1con(nodeclass);
        if flag1con==1
            disp('Network Connected!');
            break;
        % else
            % error('Network Disconnected!');
            % continue;
        end
    end
end

 function flag1con=verify1con(nodeclass)
    conmatrix=nodeclass.conmatrix;
    nodenum=size(conmatrix,1);
    spanningtree=stbfs(nodeclass);
    flag1con=(nodenum==sum(spanningtree.nodeflag));
 end

 
 function spanningtree=stbfs(nodeclass)

    conmatrix=nodeclass.conmatrix;

    nodeflag=zeros(size(conmatrix,1),1);
    nodelabel=1;
    nodeflag(nodelabel(end))=1;
    edgelabel=[];
    qset=1;

    while 1
        if isempty(qset)
            break;
        else
            parent=qset(1);
            qset(1)=[];
            temp=conmatrix(:,parent)~=0;
            temp(nodeflag==1)=0;
            todo=find(temp~=0);
            if ~isempty(todo)
                nodelabel=[nodelabel;todo];
                nodeflag(todo)=1;
                edgelabel=[edgelabel;[parent*ones(size(todo)),todo]];
                qset=[qset;todo];
            end
        end
    end

    spanningtree.nodeflag=nodeflag;
    spanningtree.nodelabel=nodelabel;
    spanningtree.edgelabel=edgelabel;
end
