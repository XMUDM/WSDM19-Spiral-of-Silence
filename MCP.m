function [out,bu,bv,U,V] = MCP(seed,trainData,testData,varargin)
%% Control random number generation
rng(seed)
%% Parse parameters
params = inputParser;
params.addParameter('method','MCP',@(x) ischar(x));
params.addParameter('m',15400,@(x) isnumeric(x));
params.addParameter('n',1000,@(x) isnumeric(x));
params.addParameter('F',5,@(x) isnumeric(x));
params.addParameter('C',2,@(x) isnumeric(x));
params.addParameter('lrT',1e-8,@(x) isnumeric(x));
params.addParameter('lr',5*1e-8,@(x) isnumeric(x));
params.addParameter('sigU',0.5,@(x) isnumeric(x));
params.addParameter('sigV',0.5,@(x) isnumeric(x));
params.addParameter('sigB',0.5,@(x) isnumeric(x));
params.addParameter('sigR',1,@(x) isnumeric(x));
params.addParameter('prior',2,@(x) isnumeric(x));
params.addParameter('maxIter',3000,@(x) isnumeric(x));
params.addParameter('topN',10,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
%% Run and evaluate model
methodSolver = str2func([par.method,'_solver']);
[bu,bv,U,V] = feval(methodSolver,trainData,testData,par);
out = MCP_eval(testData,bu,bv,U,V,par.topN);
end

function [bu,bv,U,V,Tau] = MCP_solver(trainData,testData,par)
% Initial
U = 0.1*rand(par.m,par.F);
V = 0.1*rand(par.n,par.F);
bu = 0.1*rand(par.m,1);
bv = 0.1*rand(par.n,1);
Tau = rand(par.C,1);
Beta = 0.5*ones(1,par.C);
dataMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
observeIdx = find(dataMat~=0);
missIdx = find(dataMat==0);
[~,observeCol] = ind2sub([par.m,par.n],observeIdx);
[~,missCol] = ind2sub([par.m,par.n],missIdx);
[itemSet,p] = numunique(trainData(:,2));
itemMean = arrayfun(@(x) mean(trainData(p{x},3)),(1:length(itemSet))');
% Iteration
oldLikelihood = -1e10;
fileID = fopen('MCP_out.txt','a');
for i = 1:par.maxIter
    tic;
    pred = U*V'+repmat(bu,1,par.n)+repmat(bv',par.m,1);
    % E-step 
    Omega = zeros(par.C,par.m,par.n);
    for j = 1:par.C
        temp = zeros(par.m,par.n);
        temp(observeIdx) = normpdf(dataMat(observeIdx)-pred(observeIdx),0,par.sigR).*(1./exp(Tau(j)*abs(dataMat(observeIdx)-itemMean(observeCol))));
        temp(missIdx) = 1-(1./exp(Tau(j)*abs(pred(missIdx)-itemMean(missCol))));
        Omega(j,:,:) = temp;
    end
    Pi = repmat(Beta,par.m,1);
    for j = 1:par.n
        Pi = Pi.*Omega(:,:,j)';
        Pi = Pi./sum(Pi,2);
    end
    fprintf('iter [%d/%d], update Omega completed\n',i,par.maxIter); 
    % M-step
    Beta = (sum(Pi)+par.prior-1)/sum((sum(Pi)+par.prior-1));
    fprintf('iter [%d/%d], update Beta completed\n',i,par.maxIter); 
    for j = 1:par.C 
        temp = zeros(par.m,par.n);
        temp(observeIdx) = -abs(dataMat(observeIdx)-itemMean(observeCol));
        temp(missIdx) = abs(pred(missIdx)-itemMean(missCol))./(exp(Tau(j)*abs(pred(missIdx)-itemMean(missCol)))-1);
        Tau(j) = Tau(j)+par.lrT*sum(Pi(:,j).*sum(temp,2));
    end
    fprintf('iter [%d/%d], update Tau completed\n',i,par.maxIter); 
    posIdx = missIdx((pred(missIdx)-itemMean(missCol))>=0);
    negIdx = missIdx((pred(missIdx)-itemMean(missCol))<0);
    posCol = missCol((pred(missIdx)-itemMean(missCol))>=0);
    negCol = missCol((pred(missIdx)-itemMean(missCol))<0);
    s1 = zeros(par.m,par.F);
    s2 = zeros(par.m,1);
    s3 = zeros(par.n,par.F);
    s4 = zeros(par.n,1);
    for j = 1:par.C
        temp = zeros(par.m,par.n);
        temp(observeIdx) = (dataMat(observeIdx)-pred(observeIdx))/par.sigR^2;
        temp(posIdx) = Tau(j)./((exp(Tau(j)*(pred(posIdx)-itemMean(posCol))))-1);
        temp(negIdx) = -Tau(j)./(exp(Tau(j)*(itemMean(negCol)-pred(negIdx)))-1);
        s2 = s2+Pi(:,j).*sum(temp,2);
        temp1 = V';
        temp1(:,:,ones(1,par.m));
        temp1 = permute(temp1,[3 2 1]);
        temp1 = (Pi(:,j).*temp).*temp1;
        s1 = s1+reshape(sum(temp1,2),size(temp1,1),size(temp1,3));
        s4 = s4+sum(Pi(:,j).*temp)';
        temp1 = U';
        temp1(:,:,ones(1,par.n));
        temp1 = permute(temp1,[3 2 1]);
        temp1 = (Pi(:,j).*temp)'.*temp1;
        s3 = s3+reshape(sum(temp1,2),size(temp1,1),size(temp1,3));
    end
    U = U+par.lr*(s1-U/par.sigU^2);
    bu = bu+par.lr*(s2-bu/par.sigB^2);
    V = V+par.lr*(s3-V/par.sigV^2);
    bv = bv+par.lr*(s4-bv/par.sigB^2);
    fprintf('iter [%d/%d], update U/V/bu/bv completed\n',i,par.maxIter); 
    % Log-likelihood
    likelihood = 0;
    likelihood = likelihood-sum(sum(U.^2))/(2*par.sigU^2);
    likelihood = likelihood-sum(sum(V.^2))/(2*par.sigV^2);
    likelihood = likelihood-sum(bu.^2)/(2*par.sigB^2);
    likelihood = likelihood-sum(bv.^2)/(2*par.sigB^2);
    likelihood = likelihood+sum(Pi)*log(Beta)';
    pred = U*V'+repmat(bu,1,par.n)+repmat(bv',par.m,1);
    for j = 1:par.C
        temp = zeros(par.m,par.n);
        temp(observeIdx) = -(dataMat(observeIdx)-pred(observeIdx)).^2/(2*par.sigR^2)-Tau(j)*abs(dataMat(observeIdx)-itemMean(observeCol));
        temp(missIdx) = log(exp(Tau(j)*abs(pred(missIdx)-itemMean(missCol)))-1)-Tau(j)*abs(pred(missIdx)-itemMean(missCol));
        likelihood = likelihood+sum(sum(Pi(:,j).*temp));
    end
    t = toc;
    fprintf(fileID,'iter [%d/%d], likelihood is %f, time is %f\n',i,par.maxIter,likelihood,t); 
    fprintf(fileID,'iter [%d/%d], Beta is %f/%f, Tau is %f/%f\n',i,par.maxIter,Beta,Tau); 
    % Eval
    out = MCP_eval(testData,bu,bv,U,V,par.topN);
    fprintf(fileID,'iter [%d/%d], NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',i,par.maxIter,out);
    if likelihood >= oldLikelihood && i > 2
        par.lrT = par.lrT*1.05;
        par.lr = par.lr*1.05;
    else
        par.lrT = par.lrT*0.5;
        par.lr = par.lr*0.5;
    end
    oldLikelihood = likelihood;
end
end