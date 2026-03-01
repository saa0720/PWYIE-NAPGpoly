%% data generation parameter

rng(720);

% specify the weight parameter
pi_sa = 0.3;

m = 10;



rho = 0.6; 
Sigma = zeros(m, m);


for i = 1:m
    for j = 1:m
        Sigma(i,j) = rho^abs(i-j);
    end
end


omega_true = [2 3 -1 1 0 0 0 0 0 0]';


n_train = 1000;
n_test = 10000;


K = 5;
lambdas = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001];



%% generate training dataset
T_train = mvnrnd(zeros(1, m), Sigma, n_train);
linear_pred = T_train * omega_true;
prob_train = 1 ./ (1 + exp(-linear_pred));
D_train = (rand(n_train, 1) < prob_train);
   

%% generate testing dataset
T_test = mvnrnd(zeros(1, m), Sigma, n_test);
linear_pred = T_test * omega_true;
prob_test = 1 ./ (1 + exp(-linear_pred));
D_test = (rand(n_test, 1) < prob_test);
    

%% cross-validation
random_indices = randperm(n_train);
cv_sa_test = zeros(length(lambdas), K);


for idx = 1:length(lambdas)
    for k = 1:K
        
        val_indices = random_indices(floor((k-1) * n_train / K) + 1 : floor(k * n_train / K));
        train_indices = setdiff(1:n_train, val_indices);
            
        T_cv_train = T_train(train_indices, :);
        D_cv_train = D_train(train_indices);
        T_cv_val = T_train(val_indices, :);
        D_cv_val = D_train(val_indices);
            

        num_ones = sum(D_cv_train == 1);
        num_zeros = sum(D_cv_train == 0);
        h = (num_ones * num_zeros)^(-0.1);
           
        lambda = lambdas(idx);


        params.coff = 3.7;
        regular = @(x) SCAD(x,lambda,params.coff);

        w0 = zeros((m+1),1);
            
        fns.f = @(x) weighted_youden_app(x, pi_sa, T_cv_train, D_cv_train, h, regular);
        fns.Grad = @(x) [Sn1_weighted_omega(x, pi_sa, T_cv_train, D_cv_train, h); 
                         Sn1_weighted_c(x,pi_sa,T_cv_train,D_cv_train,h)];
            
        params.u0 = w0;
        params.v0 = w0;
        params.delta = 0.1;
        params.eta = 0.1;
        params.maxiter = 100;
        params.tol = 1e-15;
        params.c1 = 0.1;
        params.ratio1 = 0.5;
        params.ratio2 = 0.99;
        params.max_ls_iter = 10;
        params.min_alpha = 1e-4;
        params.max_alpha = 1.0;
        params.lambda = lambda;
        params.linesearch_option = 1;
        params.prox = @(x,alpha) proximal_operator_SCAD(x,lambda,params.coff);
            
        [u_opt, ~, ~] = NonmonotoneAPG(fns, params);
            
 
        est_sa1 = u_opt(1:end-1);
        norm_val = norm(est_sa1);

        if norm_val > 1e-10 
            est_sa = u_opt / norm_val;
        else
            est_sa = u_opt; 
        end
            
            
        score = T_cv_val * est_sa(1:end-1);
        predict_sa = score > est_sa(end);
        youden_val = weighted_Youden_evaluate(D_cv_val, predict_sa, pi_sa);
        cv_sa_test(idx, k) = youden_val;            
    end
end


sa_test = mean(cv_sa_test, 2);
[~, max_index_sa] = max(sa_test);
lambda1 = lambdas(max_index_sa);

num_ones = sum(D_train == 1); 
num_zeros = sum(D_train == 0); 
h = (num_ones * num_zeros)^(-0.1);
num_iterations = 100;
tol = 1e-6;

lambda = lambda1;
params.u0 = w0; % 初始点
params.v0 = w0;
 
    
[u_opt, v_opt, obj_values] = NonmonotoneAPG(fns, params);

est_sa1 = u_opt(1:end-1);
norm_val = norm(est_sa1);

if norm_val > 1e-10 
    est_sa = u_opt / norm_val;
else
    est_sa = u_opt; 
    warning('Optimization returned zero vector. Skipping normalization.');
end


%the weighted youden index in training dataset
score = T_train*est_sa(1:end-1);
predict_sa = score > est_sa(end);
youden_train = weighted_Youden_evaluate(D_train, predict_sa, pi_sa);


%the weighted youden index in testing dataset
score = T_test*est_sa(1:end-1);
predict_sa = score > est_sa(end);
youden_test = weighted_Youden_evaluate(D_test, predict_sa, pi_sa);


est_sa
youden_train
youden_test





%%
function out = weighted_Youden_evaluate(D_true, D_hat, pi1)
  se = mean(D_hat(D_true==1));
  sp = mean(1-D_hat(D_true==0));
  out = (2*(pi1*se+(1-pi1)*sp)-1);
end



function out = weighted_cutpoint_selection(D_true, pred_hat, pi1) %pred_hat为估计的连续的biomarker的值
  min_value = min(pred_hat);
  max_value = max(pred_hat);
  cutoff = linspace(min_value,max_value,1000 );% seq(from=min_value, to=max_value, length.out=1000)
  n = length(pred_hat);
  D_hat = zeros(n,1);
  youden_hat = zeros(1000,1);
  for i = 1:1000
    D_hat(pred_hat<cutoff(i)) = 0;
    D_hat(pred_hat>cutoff(i)) = 1;
    youden_hat(i) = weighted_Youden_evaluate(D_true, D_hat, pi1);
  end
  [~,index_max] = max(youden_hat);
  cutpoint = cutoff(index_max);
  out = [cutpoint, max(youden_hat)];
end


function gra = Sn1_weighted_omega(parameter, pi1, Tall, Dall, h)

  p = length(parameter)-1;
  omega0 = parameter(1:p);
  c0 = parameter(p+1);
  
  %Tall_1为D=1对应的biomarker
  Tall_1 = Tall(Dall==1,:);
  %Tall_0为D=0对应的biomarker
  Tall_0 = Tall(Dall==0,:);
  
  n1 = size(Tall_1,1);
  n0 = size(Tall_0,1);
  %sa1储存第二部分的
  sa1 = zeros( p,1);
  for i = 1:n1
%     X1_sa = reshape(Tall_1(i,:), p,1);
    X1_sa = Tall_1(i,:)';
    sa = ((c0 - (omega0'* X1_sa))/h);
    sa1 = sa1 + normpdf(sa, 0, 1)*X1_sa;
  end
  %对每一列求和，再除以n1*h，得到第一部分的梯度
  sa1_1 = sa1/(n1*h);
  
  %sa0为储存第一部分的分量
  sa0 = zeros(p,1);
  for i = 1:n0
%     Y0_sa = reshape(Tall_0(i,:),p);
    Y0_sa = Tall_0(i,:)';
    sa = (c0 - (omega0'* Y0_sa))/h;
    sa0 = sa0 + normpdf(sa, 0, 1)*Y0_sa;
  
  %对每一列求和，再除以n0*h，得到第一部分的梯度
  sa0_1 = sa0/(n0*h);
  end
  %gra为梯度
  gra = (1-pi1) * sa0_1 - pi1 * sa1_1;

end


function gra = Sn1_weighted_c(parameter, pi1, Tall, Dall, h)
  p = length(parameter)-1;
  omega0 = parameter(1:p);
  c0 = parameter(p+1);
  
  %Tall_1为D=1对应的biomarker
  Tall_1 = Tall(Dall==1,: );
  %Tall_0为D=0对应的biomarker
  Tall_0 = Tall(Dall==0,: );
  
  n1 = size(Tall_1,1);
  n0 = size(Tall_0,1);
  
  %对于D=0那部分的梯度
  sa0_1 = 0;
  for i = 1:n0
    Y0_sa = Tall_0(i,:)';
    sa = ((c0 - (omega0'* Y0_sa))/h);
    sa0_1 = sa0_1 + normpdf(sa, 0, 1);
  end
  %对每一列求和，再除以n0*h，得到第一部分的梯度
  sa0_1 = sa0_1/(n0*h);
  
  %sa1为储存D=1那部分的梯度
  sa1_1 = 0;
  for i = 1:n1
    X1_sa = Tall_1(i,:)';
    sa = ((c0 - (omega0' * X1_sa))/h);
    sa1_1 = sa1_1 + normpdf(sa, 0, 1);
  end
  %对每一列求和，再除以n1*h，得到第一部分的梯度
  sa1_1 = sa1_1/(n1*h);
  
  %gra为梯度
  gra = pi1 * sa1_1 - (1-pi1) * sa0_1;
  %gra为一个数值
  
%   return(gra)
end


function out = weighted_youden_app(parameter, pi1, Tall, Dall, h,regular)
    % MATLAB implementation of youden_inverse_app using normcdf
    % INPUTS:
    % parameter - coefficients and cutoff [beta, cutoff]
    % pi1 - sensitivity的权重
    % Tall - all biomarker values
    % Dall - corresponding true status values (0 or 1)
    % h - bandwidth parameter for the approximation function
    %计算1- weighted youden
    
    % Separate parameters into beta coefficients and cutoff value
    p = length(parameter) - 1;   % Number of beta coefficients
    beta = parameter(1:p);       % Coefficients for linear combination
    cutoff = parameter(p + 1);   % Cutoff value for classification
    
    % Separate biomarker values based on true status
    Tall_1 = Tall(Dall == 1, :); % Subset of Tall where D = 1
    Tall_0 = Tall(Dall == 0, :); % Subset of Tall where D = 0
    
    % Calculate scores for D = 1 and D = 0 groups
    n_1 = size(Tall_1, 1);       % Number of samples in D = 1 group
    score_1 = Tall_1 * beta;     % Scores for D = 1
    n_0 = size(Tall_0, 1);       % Number of samples in D = 0 group
    score_0 = Tall_0 * beta;     % Scores for D = 0
    
    % Calculate indicator function values using normcdf
    indi_1 = normcdf((cutoff - score_1) / h); % For D = 1 group
    indi_0 = normcdf((cutoff - score_0) / h); % For D = 0 group
    
    % Calculate Youden's index
    youden_0 = (1-pi1) * mean(indi_0) - pi1 * mean(indi_1);
    
    % Output the value of the objective function
    out = 1 - youden_0;
    
    out = out + regular(beta);
end


function out = youden_inverse_app(parameter, Tall, Dall, h,regular)
    % MATLAB implementation of youden_inverse_app using normcdf
    % INPUTS:
    % parameter - coefficients and cutoff [beta, cutoff]
    % Tall - all biomarker values
    % Dall - corresponding true status values (0 or 1)
    % h - bandwidth parameter for the approximation function
    
    % Separate parameters into beta coefficients and cutoff value
    p = length(parameter) - 1;   % Number of beta coefficients
    beta = parameter(1:p);       % Coefficients for linear combination
    cutoff = parameter(p + 1);   % Cutoff value for classification
    
    % Separate biomarker values based on true status
    Tall_1 = Tall(Dall == 1, :); % Subset of Tall where D = 1
    Tall_0 = Tall(Dall == 0, :); % Subset of Tall where D = 0
    
    % Calculate scores for D = 1 and D = 0 groups
    n_1 = size(Tall_1, 1);       % Number of samples in D = 1 group
    score_1 = Tall_1 * beta;     % Scores for D = 1
    n_0 = size(Tall_0, 1);       % Number of samples in D = 0 group
    score_0 = Tall_0 * beta;     % Scores for D = 0
    
    % Calculate indicator function values using normcdf
    indi_1 = normcdf((cutoff - score_1) / h); % For D = 1 group
    indi_0 = normcdf((cutoff - score_0) / h); % For D = 0 group
    
    % Calculate Youden's index
    youden_0 = mean(indi_0) - mean(indi_1);
    
    % Output the value of the objective function
    out = 1 - youden_0;
    
    out = out + regular(beta);
end


function gra = Sn1_omega(parameter, Tall, Dall, h)

  p = length(parameter)-1;
  omega0 = parameter(1:p);
  c0 = parameter(p+1);
  
  %Tall_1为D=1对应的biomarker
  Tall_1 = Tall(Dall==1,:);
  %Tall_0为D=0对应的biomarker
  Tall_0 = Tall(Dall==0,:);
  
  n1 = size(Tall_1,1);
  n0 = size(Tall_0,1);
  %sa1储存第二部分的
  sa1 = zeros( p,1);
  for i = 1:n1
%     X1_sa = reshape(Tall_1(i,:), p,1);
    X1_sa = Tall_1(i,:)';
    sa = ((c0 - (omega0'* X1_sa))/h);
    sa1 = sa1 + normpdf(sa, 0, 1)*X1_sa;
  end
  %对每一列求和，再除以n1*h，得到第一部分的梯度
  sa1_1 = sa1/(n1*h);
  
  %sa0为储存第一部分的分量
  sa0 = zeros(p,1);
  for i = 1:n0
%     Y0_sa = reshape(Tall_0(i,:),p);
    Y0_sa = Tall_0(i,:)';
    sa = (c0 - (omega0'* Y0_sa))/h;
    sa0 = sa0 + normpdf(sa, 0, 1)*Y0_sa;
  
  %对每一列求和，再除以n0*h，得到第一部分的梯度
  sa0_1 = sa0/(n0*h);
  end
  %gra为梯度
  gra = sa0_1 - sa1_1;

end


function gra = Sn1_c(parameter, Tall, Dall, h)
  p = length(parameter)-1;
  omega0 = parameter(1:p);
  c0 = parameter(p+1);
  
  %Tall_1为D=1对应的biomarker
  Tall_1 = Tall(Dall==1,: );
  %Tall_0为D=0对应的biomarker
  Tall_0 = Tall(Dall==0,: );
  
  n1 = size(Tall_1,1);
  n0 = size(Tall_0,1);
  
  %对于D=0那部分的梯度
  sa0_1 = 0;
  for i = 1:n0
    Y0_sa = Tall_0(i,:)';
    sa = ((c0 - (omega0'* Y0_sa))/h);
    sa0_1 = sa0_1 + normpdf(sa, 0, 1);
  end
  %对每一列求和，再除以n0*h，得到第一部分的梯度
  sa0_1 = sa0_1/(n0*h);
  
  %sa1为储存D=1那部分的梯度
  sa1_1 = 0;
  for i = 1:n1
    X1_sa = Tall_1(i,:)';
    sa = ((c0 - (omega0' * X1_sa))/h);
    sa1_1 = sa1_1 + normpdf(sa, 0, 1);
  end
  %对每一列求和，再除以n1*h，得到第一部分的梯度
  sa1_1 = sa1_1/(n1*h);
  
  %gra为梯度
  gra = sa1_1 - sa0_1;
  %gra为一个数值
  
%   return(gra)
end


function out = SCAD(paras, alpha, coff) 
    out = 0;
    for ii = 1: length(paras) - 1
        if abs(paras(ii)) < alpha
            out = out + alpha*abs(paras(ii));
        elseif abs(paras(ii))<coff*alpha
%             alphatmp = (coff*alpha - paras)/(coff - 1) ;
            out = out + (-paras(ii)^2 + 2*alpha*coff*abs(paras(ii))-alpha^2 ) /(2*(coff -1 ));
        else
            out = out + alpha^2*(coff + 1)/2;
        end
    end
%     out(end) = paras(end);
end

