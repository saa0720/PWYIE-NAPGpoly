function [u_opt, v_opt, out] = NonmonotoneAPG(fns, params)
% NonmonotoneAPG implements the nonmonotone accelerated proximal gradient algorithm
% INPUT:
% fns: Struct containing function handles for objective function f and gradient Grad
% params: Struct containing the parameters for the algorithm
% OUTPUT:
% u_opt: Final solution u
% v_opt: Final solution v
% obj_values: Objective function values at each iteration

% Initialize parameters
uk = params.u0;
vk = params.v0;%原始向量
wk1 = params.u0;
vk1 = params.v0;
t1 = 1;
t0 = 1;
q = 1;
C = fns.f(vk);  % Initial objective function value
delta = params.delta;
obj_values = C; % Store objective values
gardnormall = [];
searchcount = [];
% Start main loop
for k = 1:params.maxiter
    % Compute w_k, s_k, and r_k
    wk = vk + t0/t1*(uk - vk) + ((t0 - 1) / t1) * (vk - vk1) ;


    sk = wk - wk1;
    rk = fns.Grad(wk) - fns.Grad(wk1);

    % Calculate alpha_y
    if  k == 1
        alpha_y = 1;
    elseif mod(k, 2) == 1
        alpha_y = abs(sk' * sk) / abs(sk' * rk);
    else
        alpha_y = abs(sk' * rk) / abs(rk' * rk);
    end
    alpha_y = max(alpha_y,1e-5);
    alpha_y = min(alpha_y, params.max_alpha); 
    % Use Algorithm 2 for line search
    if params.linesearch_option == 1
    [uk, flag,outy] = polybacktracking(fns, wk, params, alpha_y,C);
    elseif params.linesearch_option == 2
    [uk, flag,outy] = backtracking(fns, wk, params,C);
    end
    
    if flag == 0
        vk = uk; 
    else
        sk = vk - wk1;
        rk = fns.Grad(vk) - fns.Grad(uk);
        if  k == 1
            alpha_x = 1;
        elseif mod(k, 2) == 1
            alpha_x = abs(sk' * sk) / abs(sk' * rk);
        else
            alpha_x = abs(sk' * rk) / abs(rk' * rk);
        end
        alpha_x = max(alpha_x,1e-5);
        % Use Algorithm 2 again for line search
        params.lineopt = 1;

        if params.linesearch_option == 1
        [zk, ~,outx] = polybacktracking(fns, uk, params, alpha_x,C);
        elseif params.linesearch_option == 2
        [zk, flag,outx] = backtracking(fns, wk, params,C);
        end

        if outy.f_val < outx.f_val
            vk = uk;
            gradnorm = outy.gradnorm;
        else
            vk = zk;
            gradnorm = outx.gradnorm;
        end
    end
    gradw = fns.Grad(vk);
    gradnorm = norm(vk -  params.prox(vk -  gradw, params.lambda));

    if norm(gradnorm) < params.tol
        break;
    end

    % Update variables
    t0 = t1;
    t1 = (1 + sqrt(4 * t0^2 + 1)) / 2; % Update t value
    q = params.eta * q + 1;
    C = (params.eta * C + fns.f(vk)) / q; % Update C value
    obj_values = [obj_values, fns.f(vk)]; % Store objective values
    gardnormall = [gardnormall gradnorm];
    searchcount = [searchcount outy.count];
    wk1 = wk; % Update u
    vk1 = vk;

    % Check for convergence

end

% Return final solutions
u_opt = uk;
v_opt = vk;
out.obj_values = obj_values;
out.gradnormall = gardnormall;
out.searchtime = searchcount;
end
