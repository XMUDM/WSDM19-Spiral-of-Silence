function out = MCP_eval(testData, bu, bv, U, V, topN)
pred = sum(U(testData(:,1),:).*V(testData(:,2),:),2)+bu(testData(:,1))+bv(testData(:,2));
[userSet,p] = numunique(testData(:,1));
temp = zeros(length(userSet),topN);
for i = 1:length(userSet)
    for j = 1:topN
        r = (1:j)';
        if length(p{i})>r(end)
            rel = testData(p{i},3);
            [~,I] = sort(pred(p{i}),'descend');
            dcg = sum((2.^rel(I(r))-1)./log2(r+1));
            [~,I] = sort(rel,'descend');
            idcg = sum((2.^rel(I(r))-1)./log2(r+1));
            temp(i,j) = sum(dcg/idcg);
        else
            tr = (1:length(p{i}))';
            rel = testData(p{i},3);
            [~,I] = sort(pred(p{i}),'descend');
            dcg = sum((2.^rel(I(tr))-1)./log2(tr+1));
            [~,I] = sort(rel,'descend');
            idcg = sum((2.^rel(I(tr))-1)./log2(tr+1));
            temp(i,j) = sum(dcg/idcg);
        end
    end
end
out = mean(temp);
end