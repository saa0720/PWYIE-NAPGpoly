function [u_opt, flag, out] = polybacktracking(fns, u1, params, alpha_init,C)
    % Algorithm2 implements the polynomial interpolation-based backtracking algorithm
    % INPUT:
    % fns: Struct containing function handles for objective function f and gradient Grad
    % u1: Current iterate
    % params: Struct containing the parameters for the algorithm
    % alpha_init: Initial step size
    % OUTPUT:
    % u_opt: Updated iterate
    % f_val: Objective function value at the updated iterate
    % alpha_opt: Optimal step size found by line search

    % Initialize variables
    alpha = alpha_init;
    h0 = fns.f(u1);              % Initial function value h(0)
    grad_u1 = fns.Grad(u1);       % Gradient at u1
%     h0_prime = grad_u1' * (u1 - prox(u1 - alpha * grad_u1,alpha* params.lambda)); % Compute initial slope h'(0)
%     h0_prime = norm(prox(u1 - alpha * grad_u1,alpha* params.lambda),'fro')^2 ;
    i = 0;                        % Iteration counter
    alpha1 = alpha;               % Store initial step size
    flag = 0;
    fval = h0;
    % Start line search loop
    while true
        % Calculate proximal point and its function value
        u_opt = params.prox(u1 - alpha * grad_u1, alpha* params.lambda);
        h0_prime = -norm( (u1 - u_opt)/alpha)^2;
        f_val = fns.f(u_opt);     % Evaluate objective function at u_opt

        % Check sufficient decrease condition
        if f_val <= h0 + params.c1 * h0_prime * alpha || f_val <= C + params.c1 * h0_prime * alpha
            if f_val <= C - params.c1 * h0_prime * alpha
                flag = 0;
            else
                flag = 1;
            end
            alpha_opt = alpha;
            out.alpha_opt = alpha_opt;
            out.f_val = f_val;
            out.gradnorm = abs(h0_prime);
            out.count = i+1;
            return;
        elseif i == 0
            alpha1 = alpha;
        end

        % Update step size using quadratic or cubic interpolation
        if i == 0
            % Use quadratic interpolation when i = 0
            alpha = -(h0_prime * alpha1^2) / (2 * (f_val - h0 - h0_prime * alpha1));
            alpha2 = alpha;
            h_alpha1 = f_val;
        else
            % Use cubic interpolation when i > 0
%             h_alpha1 = fns.f(prox(u1 - alpha1  * grad_u1,alpha* params.lambda));            % Store function value at alpha1
            h_alpha2 = f_val; % Compute h(alpha2)
            
            % Define coefficients for cubic interpolation
            tmp1 = h_alpha1 - h0 - h0_prime * alpha1;
            tmp2 = h_alpha2 - h0 - h0_prime * alpha2;
            a = 1 / (alpha1 - alpha2) * (1/alpha1^2*tmp1 - 1/alpha2^2*tmp2);
            b = 1 / (alpha1 - alpha2) * (-alpha2 * tmp1  / alpha1^2 + (alpha1*tmp2) / (alpha2)^2);

            % Solve the cubic equation using the derived formula
            alpha = (-b + sqrt(b^2 - 3 * a * h0_prime)) / (3 * a);
            alpha1 = alpha2;
            alpha2 = alpha;
            h_alpha1 = f_val;
%             alphaim1 = alpha1;
        end

        alpha = min(max(alpha, params.ratio1*alpha1 ), params.ratio2*alpha1);

        % Update iteration counter and ensure step size is within bounds
        i = i + 1;
        if i >= params.max_ls_iter
            fprintf('Warning: Line search did not converge after %d iterations\n', i);
            alpha_opt = alpha;
            out.alpha_opt = alpha_opt;
            out.f_val = f_val;
            out.gradnorm = abs(h0_prime);
            out.count = i;
            return;
        end
    end
end

% function prox_result = prox(x, lambda)
%     % Proximal operator for L1 norm
%     prox_result = max(0, x - lambda) - max(0, -x - lambda);
% end

% function out = prox(paras,  alpha) 
%   %omega为参数，lambda为lasso的正则化参数，alpha为proximal算子中的步长
%   out = zeros(size(paras));
%   out(1:end-1) = sign(paras).* max(0, abs(paras) -  alpha);
%   out(end) = paras(end);
% end
