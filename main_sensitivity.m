% low-dimension情况下第一种数据生成方式
% logistic model但是变量间有较强相关性


rng(720);

n_sa = 100000;
m = 10;

% 生成 AR(1) 相关矩阵 (Toeplitz结构): Sigma_ij = rho^|i-j|
rho = 0.6; % 试着设高一点，0.5 到 0.8 之间
Sigma = zeros(m, m);

% 生成 AR(1) 相关矩阵 (Toeplitz结构): Sigma_ij = rho^|i-j|
for i = 1:m
    for j = 1:m
        Sigma(i,j) = rho^abs(i-j);
    end
end


omega_true = [2 3 -1 1 0 0 0 0 0 0]';
%omega_true = omega_true/norm(omega_true);


% 使用 mvnrnd 生成具有相关性的数据
T_sa = mvnrnd(zeros(1, m), Sigma, n_sa);


%% 4. 生成标签 D_sa (Logistic Model)
% 计算线性预测算子
linear_pred = T_sa * omega_true;
% 计算概率 p = 1 / (1 + exp(-x'beta))
prob_sa = 1 ./ (1 + exp(-linear_pred));


% 伯努利试验生成观测值 D (0 或 1)
D_sa = (rand(n_sa, 1) < prob_sa);

%% 5. 计算真实的 Cutpoint (基于你现有的函数)
pi_sa = 0.3;
cutpoint_sa = weighted_cutpoint_selection(D_sa, T_sa * omega_true, pi_sa);


n_train = 400;
%num_sa为生成的数据的次数
num_sa = 300;
pi_values = 0.1:0.1:0.9;  % 9个 pi 值
n_pi = length(pi_values);



%交叉验证参数
K = 5;
lambdas = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001];


% 存储结果：n_pi x m，每行是某个 pi 下 m 个系数的平均估计值
coef_mean = zeros(n_pi, m);
coef_std  = zeros(n_pi, m);  % 也保存标准差，方便后续加误差带


%存储每次选择的lambda
selected_lambdas = zeros(num_sa, 2);

%%
for pi_idx = 1:n_pi

    pi_sa = pi_values(pi_idx);
    fprintf('\n===== pi = %.1f (%d/%d) =====\n', pi_sa, pi_idx, n_pi);

    % 存储本 pi 值下 num_rep 次的系数估计
    coef_rep = zeros(num_sa, m);

    for iter = 1:num_sa
   
        fprintf('  pi=%.1f  iter %d/%d\n', pi_sa, iter, num_sa);
    
        %%
        % 生成训练集数据
        T_train = mvnrnd(zeros(1, m), Sigma, n_train);
        linear_pred = T_train * omega_true;
        prob_train = 1 ./ (1 + exp(-linear_pred));
        D_train = (rand(n_train, 1) < prob_train);


        %%
        % 第一步：用lasso-logistic做暖启动初始化
        cv_logit_init = zeros(length(lambdas), K);
        random_indices_logit = randperm(n_train);

        %% 交叉验证选择lambda

        %生成随机索引用于CV
        random_indices = randperm(n_train);
        % 存储CV结果
        cv_sa_test = zeros(length(lambdas), K);  
        cv_logit_test = zeros(length(lambdas), K);
        
        for idx = 1:length(lambdas)
            for k = 1:K
                % 划分训练集和验证集
                val_indices = random_indices(floor((k-1) * n_train / K) + 1 : floor(k * n_train / K));
                train_indices = setdiff(1:n_train, val_indices);
                
                T_cv_train = T_train(train_indices, :);
                D_cv_train = D_train(train_indices);
                T_cv_val = T_train(val_indices, :);
                D_cv_val = D_train(val_indices);
                
                % 计算h值
                num_ones = sum(D_cv_train == 1);
                num_zeros = sum(D_cv_train == 0);
                h = (num_ones * num_zeros)^(-0.1);
                
                w0 = zeros((m+1),1);
                lambda = lambdas(idx);
                

                %% 方法1: lasso-logistic 方法
          
                [B_logit, FitInfo] = lassoglm(T_cv_train, D_cv_train, 'binomial', 'Lambda', lambda/length(train_indices));
                u_opt = B_logit;
                
                c_sa = weighted_cutpoint_selection(D_cv_train, T_cv_train*u_opt, pi_sa);
                
                if norm(u_opt) > 0
                    est_sa = [u_opt; c_sa(1)]/norm(u_opt);
                else
                    est_sa = [u_opt; c_sa(1)];
                end
                
                est_logit = est_sa;

                score = T_cv_val * est_sa(1:end-1);
                predict_sa = score > est_sa(end);
                youden_val = weighted_Youden_evaluate(D_cv_val, predict_sa, pi_sa);
                cv_logit_test(idx, k) = youden_val;


                %% 方法2: Weighted Youden (我们的方法)

                w0 = est_logit;
                params.coff = 3.7;
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
                
                regular = @(x) SCAD(x,lambda,params.coff);
                
                fns.f = @(x) weighted_youden_app(x, pi_sa, T_cv_train, D_cv_train, h, regular);
                fns.Grad = @(x) [Sn1_weighted_omega(x, pi_sa, T_cv_train, D_cv_train, h); 
                                 Sn1_weighted_c(x,pi_sa,T_cv_train,D_cv_train,h)];
                
                [u_opt, ~, ~] = NonmonotoneAPG(fns, params);
                
                % 归一化
                est_sa1 = u_opt(1:end-1);
                norm_val = norm(est_sa1);
    
                if norm_val > 1e-10 % 只有模长足够大才归一化
                    est_sa = u_opt / norm_val;
                else
                    % 如果是全零解，保持原样（全0），或者给一个默认方向
                    % 这种情况通常意味着 Lambda 太大了，或者算法没找到方向
                    est_sa = u_opt; 
                    warning('Optimization returned zero vector. Skipping normalization.');
                end
                
                % 在验证集上评估
                score = T_cv_val * est_sa(1:end-1);
                predict_sa = score > est_sa(end);
                youden_val = weighted_Youden_evaluate(D_cv_val, predict_sa, pi_sa);
                cv_sa_test(idx, k) = youden_val;        
            end
        end
    
        % 选择最优lambda
        sa_test = mean(cv_sa_test, 2);
        [~, max_index_sa] = max(sa_test);
        lambda1 = lambdas(max_index_sa);


        logit_test = mean(cv_logit_test, 2);
        [~, max_index_logit] = max(logit_test);
        lambda2 = lambdas(max_index_logit);
    
    
        % 存储选择的lambda
        selected_lambdas(iter, :) = [lambda1, lambda2];

        %%
        % 参数初始化
        num_ones = sum(D_train == 1); % 计算 D_train 中1的个数
        num_zeros = sum(D_train == 0); % 计算 D_train 中0的个数
        % 计算 h 值
        h = (num_ones * num_zeros)^(-0.1);
       

        %% lasso-logistic 方法
        lambda = lambda2;

        [B_logit, FitInfo] = lassoglm(T_train, D_train, 'binomial', 'Lambda', lambda/n_train);
     
        intercept_logit = FitInfo.Intercept;
        u_opt = B_logit;
        
        % 计算cutoff值
        c_sa = weighted_cutpoint_selection(D_train, T_train*u_opt, pi_sa);
        
        % 参数归一化
        if norm(u_opt) > 0
            est_sa = [u_opt; c_sa(1)]/norm(u_opt);
        else
            est_sa = [u_opt; c_sa(1)];
        end

        est_logit = est_sa;
        
       
        %% 我们的方法进行参数初始化
        w0 = est_logit;
        lambda = lambda1;
        params.coff = 3.7;
        regular = @(x) SCAD(x,lambda,params.coff);
       
       
        fns.f = @(x) weighted_youden_app(x, pi_sa, T_train,D_train,h,regular); % 示例目标函数
        fns.Grad = @(x) [Sn1_weighted_omega(x,pi_sa,T_train,D_train,h); Sn1_weighted_c(x,pi_sa,T_train,D_train,h)]; % 示例梯度
        
        
        params.linesearch_option = 1;
        params.u0 = w0; % 初始点
        params.v0 = w0;
        params.delta = 0.1;
        
        params.eta = 0.1;
        params.maxiter = 100;
        params.tol = 1e-15;
        params.c1 = 0.1;
        params.ratio1= 0.5;
        params.ratio2= 0.99;
        params.max_ls_iter = 10;
        params.min_alpha = 1e-4;
        params.max_alpha = 1.0;
        params.lambda = lambda1;
        % params.prox = @(x,alpha) proximal_operator_lasso(x,lambda,alpha);
        params.prox = @(x,alpha) proximal_operator_SCAD(x,lambda,params.coff) ;
    
    
        %%
    
        % 运行算法
       
        [u_opt, v_opt, obj_values] = NonmonotoneAPG(fns, params);

        %est_sa为优化后的参数归一化
        est_sa1 = u_opt(1:end-1);
        norm_val = norm(est_sa1);
    
        if norm_val > 1e-10 % 只有模长足够大才归一化
            est_sa = u_opt / norm_val;
        else
                    % 如果是全零解，保持原样（全0），或者给一个默认方向
                    % 这种情况通常意味着 Lambda 太大了，或者算法没找到方向
            est_sa = u_opt; 
            warning('Optimization returned zero vector. Skipping normalization.');
        end

        coef_rep(iter, :) = est_sa(1:m)';
    end

    % 去掉含 NaN 的行后取均值和标准差
    coef_rep = coef_rep(~any(isnan(coef_rep), 2), :);
    coef_mean(pi_idx, :) = mean(coef_rep, 1);
    coef_std(pi_idx, :)  = std(coef_rep, 0, 1);
end




pi_col = pi_values(:);
coef_names = arrayfun(@(i) sprintf('beta%d', i), 1:m, 'UniformOutput', false);

T_mean = array2table([pi_col, coef_mean], 'VariableNames', [{'pi'}, coef_names]);
writetable(T_mean, 'sensitivity_coef_mean.csv');

T_std = array2table([pi_col, coef_std], 'VariableNames', [{'pi'}, coef_names]);
writetable(T_std, 'sensitivity_coef_std.csv');

fprintf('\n结果已保存完毕\n');





true_vals = [omega_true'/norm(omega_true), cutpoint_sa(1)/norm(omega_true), cutpoint_sa(2), cutpoint_sa(2)]
result_sa = result_sa(~any(isnan(result_sa), 2), :);
mean_sa = mean(result_sa)
result_logit = result_logit(~any(isnan(result_logit), 2), :);
mean_logit = mean(result_logit)
mean_wang = mean(result_wang)
result_salaroli = result_salaroli(~any(isnan(result_salaroli), 2), :);
mean_salaroli = mean(result_salaroli)
selected_lambdas;



bias_sa = mean_sa-true_vals


sd_sa = std(result_sa)
%sd_logit = std(result_logit)




%计算RMSE
rmse_sa = sqrt(mean((result_sa - true_vals).^2))
%rmse_logit = sqrt(mean((result_logit - true_vals).^2))



%% 导出画图数据
% 假设你跑完了 simulation，有以下数据 (各 500 行)
% result_sa, result_wang, result_logit, result_salaroli

% 1. 提取你要画的那一列指标 (分别是第 12、13 列  train wYI Test wYI)
% 请根据你的 result 矩阵实际结构修改列号
% 1. 提取数据
col_idx = 12:13; 
% 这里的 y_sa 是 N x 2 的矩阵
y_sa = result_sa(:, col_idx);
y_wang = result_wang(:, col_idx);
y_logit = result_logit(:, col_idx);
y_sal = result_salaroli(:, col_idx);

% 2. 准备数据块 (分别处理 Train 和 Test)
% --- 训练集数据块 ---
Y_Train = [y_sa(:,1); y_wang(:,1); y_logit(:,1); y_sal(:,1)];
Method_Train = [repmat("Proposed", size(y_sa,1), 1); ...
                repmat("Wang(2025)", size(y_wang,1), 1); ...
                repmat("Lasso-Log", size(y_logit,1), 1); ...
                repmat("Salaroli", size(y_sal,1), 1)];
Type_Train = repmat("Train", length(Y_Train), 1); % 标签全是 "Train"

% --- 测试集数据块 ---
Y_Test = [y_sa(:,2); y_wang(:,2); y_logit(:,2); y_sal(:,2)];
Method_Test = Method_Train; % 方法标签是一样的
Type_Test = repmat("Test", length(Y_Test), 1);   % 标签全是 "Test"

% 3. 纵向堆叠 (把训练集和测试集拼起来)
Final_Y = [Y_Train; Y_Test];
Final_Method = [Method_Train; Method_Test];
Final_Type = [Type_Train; Type_Test];

% 4. 创建 Table
T_Long = table(Final_Method, Final_Type, Final_Y, ...
    'VariableNames', {'Method', 'Type', 'wYI'});

% 5. 导出
writetable(T_Long, 'model1_3.csv');





%计算置信区间
% 计算标准误差 (SE)
%n_1 = size(result_sa, 1);
%se_sa = sd_sa / sqrt(n_1);
% 设置置信水平 (alpha = 0.05 对应 95% 置信区间)
%alpha = 0.05;
% 计算 正态分布 的临界值 (根据样本量 N - 1)
%z = norminv(1 - alpha/2);
% 计算每一列的置信区间
%ci_lower = mean_sa - z * se_sa
%ci_upper = mean_sa + z * se_sa

%n_2 = size(result_logit, 1);
%se_logit = sd_logit / sqrt(n_2);
%ci_lower = mean_logit - z * se_logit
%ci_upper = mean_logit + z * se_logit




%计算每一次估计压缩到0的个数
selected_sa = result_sa(:, [5, 6, 7, 8, 9, 10]);
selected_logit = result_logit(:, [5, 6, 7, 8, 9, 10]);
selected_wang = result_wang(:, [5, 6, 7, 8, 9, 10]);
selected_salaroli = result_salaroli(:, [5, 6, 7, 8, 9, 10]);
% 对这 5 个维度中的每一行，计算值为 0 的个数
num_zeros = mean(sum(abs(selected_sa) < 1e-4, 2))/6
num_zeros_logit = mean(sum(abs(selected_logit) < 1e-4, 2))/6
num_zeros_salaroli = mean(sum(abs(selected_salaroli) < 1e-4, 2))/6
num_zeros_wang = mean(sum(abs(selected_wang) < 1e-4, 2))/6




%计算非0的
selected_sa = result_sa(:, [1,2,3,4]);
selected_logit = result_logit(:, [1,2,3,4]);
selected_wang = result_wang(:, [1,2,3,4]);
selected_salaroli = result_salaroli(:, [1,2,3,4]);
% 对这 5 个维度中的每一行，计算值为 0 的个数
num_zeros = mean(sum(abs(selected_sa) > 1e-4, 2))/4
num_zeros_logit = mean(sum(abs(selected_logit) > 1e-4, 2))/4
num_zeros_salaroli = mean(sum(abs(selected_salaroli) > 1e-4, 2))/4
num_zeros_wang = mean(sum(abs(selected_wang) > 1e-4, 2))/4




num_zeros = mean(sum(abs(result_sa) < 1e-4, 2))/10
num_zeros_logit = mean(sum(abs(result_logit) < 1e-4, 2))/10
num_zeros_salaroli = mean(sum(abs(result_salaroli) < 1e-4, 2))/10
num_zeros_wang = mean(sum(abs(result_wang) < 1e-4, 2))/10

%selected_sa = result_sa(:, [1, 3, 6, 8]);
%selected_logit = result_logit(:, [1, 3, 6, 8]);
% 对这 5 个维度中的每一行，计算值为 0 的个数
%num_zeros = 1-mean(sum(abs(selected_sa) < 0.01, 2))/4
%num_zeros_logit = 1-mean(sum(abs(selected_logit) < 0.01, 2))/4
%num_zeros = 1-mean(sum(abs(selected_sa) == 0, 2))/4
%num_zeros_logit = 1-mean(sum(selected_logit == 0, 2))/4




%num_zeros_logit = mean(sum(excluded_logit == 0, 2))/495
%num_zeros_sa = mean(sum(excluded_sa == 0, 2))/495
















    %disp('Optimal solution:');
    %disp(u_opt);
%% plot
% semilogy(out.f_val-min(obj_values))
% hold on
% semilogy(obj_values-min(obj_values)+1e-8)
%plot(out.f_val,'r-','LineWidth',2)
%hold on
%plot(obj_values,'b:','LineWidth',2)
%legend('Proxiaml gradient','NAPG-poly')
%xlabel('Iteration')
%ylabel('Function value')
1;
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

