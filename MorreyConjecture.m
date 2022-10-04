clc
clear all
format long

%potential problems in the code:
%1. Find better approximation on the derivatives of boundary points
%2. 

%1. Mesh nodes coordinates generation
%%
num_div = 10;
dis_div = 1/num_div;
nodes = FEMNodeGenerator(num_div);
gamma = 4 / sqrt(3);
%eps = [1, 0; 0 sqrt(3)];
%eps = rand(2, 2);
%eps = [0 -1; 1 0];
%eps = 0.1 * [1, 0; 0 sqrt(3)];
%eps = [0 0; 0 0];
%eps = [1, 0; 0 sqrt(2.8)];
%%

%2. Define phi0, lipschitz and vanishes at the boundaries
%%
syms x y
%phi0 = @(x,y) [sin(x + y) cos(x + y)];
%phi0 = @(x,y)  [sin(pi*x), sin(pi*y)];
%phi0 = @(x,y) [0, sin(pi*y)];
%phi0 = @(x,y) [0.01 * sin(pi * x), 0.01 * sin(3 * pi / 2 * y)];
%phi0 = @(x,y) [cos(2*pi*x), cos(2*pi*y)];
%phi0 = @(x,y) [0 0];
%phi0 = @(x,y) 1/(2 * pi) * [sin(2 * pi*x), sin(2 * pi*y)];
%phi0 = @(x,y) [sin(x), sin(x+y)];
phi0 = @(x,y) [sin(x*(x-1)*y*(y-1)), sin(x*(x-1)*y*(y-1))^2];
%phi0 = @(x,y) [(x*(x-1)*y*(y-1)), (x*(x-1)*y*(y-1))^2];
%%

%3. Value of phi0 at each node initially
%%
nodes_value = phi0_value(nodes);
%nodes_value = process_value(nodes_value);

%%
%4. Gradient of function phi0 at those nodes / Iterations start here
max_loop = 1000;

eps_array = cell(max_loop, 1);

for k = 1:max_loop

    eps = rand(2, 2);
    eps_array{k} = eps;
    gamma_array(k) = gamma;

    nodes_value = process_value(nodes_value);

    %functionalJ at phi_k which is also the updating factor
    grad1phi1 = grad1Phi(nodes_value, num_div, 1);
    grad2phi1 = grad2Phi(nodes_value, num_div, 1);
    grad1phi2 = grad1Phi(nodes_value, num_div, 2);
    grad2phi2 = grad2Phi(nodes_value, num_div, 2);
    grad11phi1 = grad11Phi(nodes_value, num_div, 1);
    grad11phi2 = grad11Phi(nodes_value, num_div, 2);
    grad22phi1 = grad22Phi(nodes_value, num_div, 1);
    grad22phi2 = grad22Phi(nodes_value, num_div, 2);
    grad12phi1 = grad12Phi(grad1phi1, num_div);
    grad12phi2 = grad12Phi(grad1phi2, num_div);
    grad21phi1 = grad21Phi(grad2phi1, num_div);
    grad21phi2 = grad21Phi(grad2phi2, num_div);

    %%determinant of (eps + grad_phi) at all nodes
    determinant = (eps(1,1) + grad1phi1) .* (eps(2,2) + grad2phi2) - (eps(1,2) + grad2phi1) .* (eps(2,1) + grad1phi2);
    %%||eps + grad_phi||^2 , norm^2
    norm = (eps(1,1) + grad1phi1).^2 + (eps(1,2) + grad2phi1).^2 + (eps(2,1) + grad1phi2).^2 + (eps(2,2) + grad2phi2).^2;
    
    %%partial wrt x of norm^2
    partialXnorm2 = 2 * (eps(1,1) + grad1phi1) .*  grad11phi1 + 2 * (eps(1,2) + grad2phi1) .*  grad21phi1 + 2 * (eps(2,1) + grad1phi2) .*  grad11phi2 + 2 * (eps(2,2) + grad2phi2) .*  grad21phi2;
    
    %%partial wrt y of norm^2
    partialYnorm2 = 2 * (eps(1,1) + grad1phi1) .*  grad12phi1 + 2 * (eps(1,2) + grad2phi1) .*  grad22phi1 + 2 * (eps(2,1) + grad1phi2) .*  grad12phi2 + 2 * (eps(2,2) + grad2phi2) .*  grad22phi2;
    
    %%ggrad_1dir_1 = ggrad_1dir_11 + ggrad_1dir_22
    %%ggrad_1dir_1 -- page 8
    ggrad_1dir_11 = 4 * (grad11phi1 .* norm + (eps(1,1) + grad1phi1) .* partialXnorm2) - ...
                   gamma * (2 * grad11phi1 .* determinant + 2 * (eps(1,1) + grad1phi1) .* ...
                   (grad11phi1.* (eps(2,2) + grad2phi2) + grad21phi2 .* (eps(1,1) + grad1phi1) ...
                   - grad11phi2 .* (eps(1,2) + grad2phi1) - grad21phi1 .* (eps(2,1) + grad1phi2))...
                   + grad21phi2 .* norm + (eps(2,2) + grad2phi2) .* partialXnorm2);
    
    %%ggrad_1dir_22 -- page 10
    ggrad_1dir_22 = 4 * (grad22phi1 .* norm + (eps(1,2) + grad2phi1) .* partialYnorm2) - ...
                    gamma * (2 * grad22phi1 .* determinant + 2 * (eps(1,2) + grad2phi1) .* ...
                    (grad12phi1 .* (eps(2,2) + grad2phi2) + grad22phi2 .* (eps(1,1) + grad1phi1) ...
                    - grad12phi2 .* (eps(1,2) + grad2phi1) - grad22phi1 .* (eps(2,1) + grad1phi2)) ...
                    - grad12phi2 .* norm + (eps(2,1) + grad1phi1) .* partialYnorm2);
    
    %%ggrad_2dir_11 -- page 17
    ggrad_2dir_11 = 4 * (grad11phi2 .* norm + (eps(2,1) + grad1phi2) .* partialXnorm2) -...
                    gamma * (2 * grad11phi2 .* determinant + 2 * (eps(2,1) + grad1phi2) .* ...
                    (grad11phi1.* (eps(2,2) + grad2phi2) + grad21phi2 .* (eps(1,1) + grad1phi1) ...
                    - grad11phi2 .* (eps(1,2) + grad2phi1) - grad21phi1 .* (eps(2,1) + grad1phi2))...
                    - grad21phi1 .* norm - (eps(1,2) + grad2phi1) .* partialXnorm2);
    
    %%ggrad_2dir_22 -- page 16
    ggrad_2dir_22 = 4 * (grad22phi2 .* norm + (eps(2,2) + grad2phi2) .* partialYnorm2) - ...
                    gamma * (2 * grad22phi2 .* determinant + 2 * (eps(2,2) + grad2phi2) .* ...
                    (grad12phi1 .* (eps(2,2) + grad2phi2) + grad22phi2 .* (eps(1,1) + grad1phi1) ...
                    - grad12phi2 .* (eps(1,2) + grad2phi1) - grad22phi1 .* (eps(2,1) + grad1phi2)) ...
                    + grad12phi1 .* norm + (eps(1,1) + grad1phi1) .* partialYnorm2);

    %%the functional of J_gamma at phi_k
    updatePhi1 = ggrad_1dir_11 + ggrad_1dir_22;
    updatePhi2 = ggrad_2dir_11 + ggrad_2dir_22;

    functionalJ_k = [-updatePhi1 -updatePhi2];

    % Secant Method to find optimal tau
    alpha_0 = 0.001; % alpha_(n-1)
    alpha_1 = 0.002; % alpha_n

    err = abs(alpha_1 - alpha_0);

    while  err >= 10^(-6)
        g_0 = inner(nodes_value, functionalJ_k, alpha_0, num_div, eps, gamma); %%n-1
        g_1 = inner(nodes_value, functionalJ_k, alpha_1, num_div, eps, gamma); %%n
        
        alpha_2 = alpha_1 - g_1 * (alpha_1 - alpha_0) / (g_1 - g_0); %%n+1

        alpha_0 = alpha_1;
        alpha_1 = alpha_2;

        err = abs(alpha_1 - alpha_0);
    end

    tau = alpha_2;

    tauPlot(k) = tau;

    %find nodes_value at phi_k+1
    nodes_value = nodes_value - tau * functionalJ_k;

    %%check Jensen Ineq
    area = 1;
    %%f(eps)
    f_eps = (eps(1,1) ^2 + eps(1,2) ^2 + eps(2,1) ^2 + eps(2,2) ^2)^2 ...
            - gamma * (eps(1,1) ^2 + eps(1,2) ^2 + eps(2,1) ^2 + eps(2,2) ^2) ...
            *(eps(1,1) * eps(2,2) - eps(1,2) * eps(2,1));
    constJen = f_eps * area;

    %%left hand side
    f_epsphi = dis_div^2 .* (norm .^2 - gamma .* norm .* determinant);
    jensenGrad = sum(f_epsphi, 'all');

    jensenPlot(k) = jensenGrad;

    difference(k) = jensenGrad - constJen;
    %%checking quasiconvex
    if (jensenGrad >= constJen)
        disp("Tau: " + tau)
        disp("Not denied: " + gamma)
        disp("Difference: " + difference(k))
        gamma = gamma - (4/sqrt(3)-2)/max_loop;
    else
        disp("Tau: " + tau)
        disp("Denied: " + gamma)
        disp("Difference: " + difference(k))
        gamma = gamma - (4/sqrt(3)-2)/max_loop;
    end
end

iteration = linspace(1,max_loop,max_loop);

fig1 = figure(1);
hold on
h(1) = plot(iteration, difference, ...
    'LineWidth',1)
title('Steepest Descent on J_\gamma(\phi^k)')
ax = gca;
ax.FontSize = 13;

xlabel('Iterations') 
ylabel('d_\gamma(\phi^k)')
ax = gca;
ax.FontSize = 15;
grid on

for i = 1:max_loop
    if difference(i) < 0
        h(2)=plot(i, difference(i),'r*')
        disp("Gamma: " + gamma_array(i))
        disp("Index: " + i)
        disp("Jensen Integral: " + jensenPlot(i))
        disp("Difference: " + difference(i))
        disp("Eps: " + eps_array{i})    
    end
end

legend(h(2),'d_\gamma(\phi)<0')
hold off


%Functions
%% phi0 value function
function y=phi0_value(x)
    %y = [sin(x(:,1) + x(:,2)) cos(x(:,1) + x(:,2))];
    %y = [sin(pi * x(:,1)) sin(pi * x(:,2))];
    %y = [sin(0 * x(:,1)) sin(pi * x(:,2))];
    y = 0.01 * [sin(pi * x(:,1)) sin(3 * pi / 2 * x(:,2))]; %this one worked
    %y = [cos(2*pi * x(:,1)) cos(2*pi * x(:,2))];
    %y = [sin(0 * x(:,1)) sin(0 * x(:,2))];
    %y = 1/(2 * pi) * [sin(2 * pi * x(:,1)) sin(2 * pi * x(:,2))]; %this one worked but all positive
    %y = [sin(x(:,1)) sin(x(:,1) + x(:,2))];
    %y = [sin(x(:,1).*(x(:,1)-1).*x(:,2).*(x(:,2)-1)) sin(x(:,1).*(x(:,1)-1).*x(:,2).*(x(:,2)-1)).^2]; %this one worked
    %y = [(x(:,1).*(x(:,1)-1).*x(:,2).*(x(:,2)-1)) sin(0 * x(:,1))]; %this one worked
end

%% Mesh coords generation

function [coords]=FEMNodeGenerator(n)
    h = 1 / n;
    [x, y] = meshgrid(0:h:1, 0:h:1);
    coords = [x(:) y(:)];
end

%% Grad1Phi1 and Grad1Phi2
function grad_1_phi = grad1Phi(nodes_value, n, f)
    h = 1/n;
    %shift the rows of nodes_value up by n+1
    nodes_value_shift = circshift(nodes_value,-n-1,1);

    %The a-th row and the a+n+1 th row have the same y coordinates
    % (f(x+h, y) - f(x,y)) / h
    grad_1_phi = (nodes_value_shift(:,f) - nodes_value(:,f)) / h;

    %Take care of the last n+1 rows
    %Generate new extra nodes to the right of the boundary
    [x y] = meshgrid(1+h, 0:h:1);
    extra_coords = [x(:) y(:)];
    extra_values = phi0_value(extra_coords);
    last_col_grad = (extra_values(:,f) - nodes_value([(n * (n+1) + 1):(n+1)^2],f)) / h;
    grad_1_phi((n * (n+1) + 1):(n+1)^2) = last_col_grad;
end

%% Grad2Phi1 and Grad2Phi2

function grad_2_phi = grad2Phi(nodes_value, n, f)
    h = 1/n;

    [x y] = meshgrid(0:h:1, 1+h);
    extra_coords = [x(:) y(:)];
    extra_values = phi0_value(extra_coords);

    for i = 1 : (n+1)^2
        if(rem(i,(n+1)) == 0)
            j = i / (n+1);
            grad_2_phi(i) = (extra_values(j, f) - nodes_value(i, f)) / h;
        else
            grad_2_phi(i) = ( (nodes_value(i+1,f)) - nodes_value(i,f)) / h;
        end
    end
    grad_2_phi = grad_2_phi';
end

%% Grad11_phi1 and Grad11_phi2
function grad_11_phi = grad11Phi(nodes_value, n, f)
    % f indicates if it's function 1 or funtion 2
    h = 1/n;
    %Need to take care of the low and top bound
    %low bound
    [x1 y1] = meshgrid(-h, 0:h:1);
    extra_coords_low = [x1(:) y1(:)];
    extra_values_low = phi0_value(extra_coords_low);
    %top bound
    [x2 y2] = meshgrid(1+h, 0:h:1);
    extra_coords_top = [x2(:) y2(:)];
    extra_values_top = phi0_value(extra_coords_top);

    %Second Gradient
    %shift the rows of grad_value up by n+1
    %f(x+h, y)
    nodes_value_shift_up = circshift(nodes_value,-n-1);
    %f(x-h, y)
    nodes_value_shift_down  = circshift(nodes_value,n+1);

    grad_11_phi = ( nodes_value_shift_up(:, f) - 2 * nodes_value(:, f) + nodes_value_shift_down(:, f)) / h^2;

    %First n+1 rows
    grad_11_phi(1:n+1) = ( nodes_value_shift_up(1:n+1, f) - 2 * nodes_value(1:n+1, f) + extra_values_low(:,f) ) / h^2;
    %Last n+1 rows
    grad_11_phi((n * (n+1) + 1):(n+1)^2) = (extra_values_top(:,f) - 2 * nodes_value((n * (n+1) + 1):(n+1)^2, f) + nodes_value_shift_down((n * (n+1) + 1):(n+1)^2, f))/ h^2; 
end

%% Grad22_phi1 and Grad22_phi2
% g_yy = ( g(x, y+h) - 2 * g(x,y) + g(x, y-h)) / h^2;
function grad_22_phi = grad22Phi(nodes_value, n, f)
    h = 1/n;
    %Need to take care of the low and top bound
    %low bound
    [x1 y1] = meshgrid( 0:h:1, -h);
    extra_coords_low = [x1(:) y1(:)];
    extra_values_low = phi0_value(extra_coords_low);
    %top bound
    [x2 y2] = meshgrid(0:h:1, 1+h);
    extra_coords_top = [x2(:) y2(:)];
    extra_values_top = phi0_value(extra_coords_top);

    for i = 1 : (n+1)^2
        if rem(i, n+1) == 1
            j = fix(i/(n+1)) + 1;
            grad_22_phi(i) = (nodes_value(i+1,f) - 2 * nodes_value(i,f) + extra_values_low(j,f)) / h^2;
        elseif rem(i, n+1) == 0
            j = i / (n+1);
            grad_22_phi(i) = (extra_values_top(j,f) - 2 * nodes_value(i,f) + nodes_value(i-1,f)) / h^2;
        else
            grad_22_phi(i) = (nodes_value(i+1,f) - 2 * nodes_value(i,f) + nodes_value(i-1,f)) / h^2;
        end
    end
    grad_22_phi = grad_22_phi';
end

%% Grad12_phi1 and Grad12_phi2
% Grad12_phi1 is the gradient of grad1_phi1 w.r.t variable 2
% (f(x, y+h) - 2 * f(x, y) + f(x, y-h)) / h^2
function grad_12_phi = grad12Phi(grad_values, n)
    h = 1/n;
    for i = 1:(1+n)^2
        if rem(i, n+1) == 0
            grad_12_phi(i) = (grad_values(i-1) - grad_values(i)) / h;
        else
            grad_12_phi(i) = (grad_values(i+1) - grad_values(i)) / h;
        end
    end

    grad_12_phi = grad_12_phi';
end

%% Grad21_phi1 and Grad21_phi2
% input is grad2_phi1
function grad_21_phi = grad21Phi(grad_values, n)
    h = 1/n;
    grad_value_shift = circshift(grad_values,-n-1);

    grad_21_phi  = (grad_value_shift - grad_values) / h;
end

%% Set boundary values to 0
function processedValue = process_value(nodes_value)
    n = sqrt(length(nodes_value));
    for i = 1:n %Left boundary
        nodes_value(i,:) = 0;
    end

    for i = n^2 - n + 1 : n^2 %Right boundary
        nodes_value(i,:) = 0;
    end

    for i = 1 : n^2
        if rem(i, n) == 1 %Bottom boundary
            nodes_value(i,:) = 0;
        elseif rem(i, n) == 0 %Top boundary
            nodes_value(i,:) = 0;
        end        
    end
    processedValue = nodes_value;
end

%% function to calculate functional_J, double sum
function update = functionalJ(nodes_value, num_div, eps, ggamma)
    %functionalJ at phi_k which is also the updating factor
    grad1phi1 = grad1Phi(nodes_value, num_div, 1);
    grad2phi1 = grad2Phi(nodes_value, num_div, 1);
    grad1phi2 = grad1Phi(nodes_value, num_div, 2);
    grad2phi2 = grad2Phi(nodes_value, num_div, 2);
    grad11phi1 = grad11Phi(nodes_value, num_div, 1);
    grad11phi2 = grad11Phi(nodes_value, num_div, 2);
    grad22phi1 = grad22Phi(nodes_value, num_div, 1);
    grad22phi2 = grad22Phi(nodes_value, num_div, 2);
    grad12phi1 = grad12Phi(grad1phi1, num_div);
    grad12phi2 = grad12Phi(grad1phi2, num_div);
    grad21phi1 = grad21Phi(grad2phi1, num_div);
    grad21phi2 = grad21Phi(grad2phi2, num_div);

    %%determinant of (eps + grad_phi) at all nodes
    determinant = (eps(1,1) + grad1phi1) .* (eps(2,2) + grad2phi2) - (eps(1,2) + grad2phi1) .* (eps(2,1) + grad1phi2);
    %%||eps + grad_phi||^2 , norm^2
    norm = (eps(1,1) + grad1phi1).^2 + (eps(1,2) + grad2phi1).^2 + (eps(2,1) + grad1phi2).^2 + (eps(2,2) + grad2phi2).^2;
    
    %%partial wrt x of norm^2
    partialXnorm2 = 2 * (eps(1,1) + grad1phi1) .*  grad11phi1 + 2 * (eps(1,2) + grad2phi1) .*  grad21phi1 + 2 * (eps(2,1) + grad1phi2) .*  grad11phi2 + 2 * (eps(2,2) + grad2phi2) .*  grad21phi2;
    
    %%partial wrt y of norm^2
    partialYnorm2 = 2 * (eps(1,1) + grad1phi1) .*  grad12phi1 + 2 * (eps(1,2) + grad2phi1) .*  grad22phi1 + 2 * (eps(2,1) + grad1phi2) .*  grad12phi2 + 2 * (eps(2,2) + grad2phi2) .*  grad22phi2;
    
    %%ggrad_1dir_1 = ggrad_1dir_11 + ggrad_1dir_22
    %%ggrad_1dir_1 -- page 8
    ggrad_1dir_11 = 4 * (grad11phi1 .* norm + (eps(1,1) + grad1phi1) .* partialXnorm2) - ...
                   ggamma * (2 * grad11phi1 .* determinant + 2 * (eps(1,1) + grad1phi1) .* ...
                   (grad11phi1.* (eps(2,2) + grad2phi2) + grad21phi2 .* (eps(1,1) + grad1phi1) ...
                   - grad11phi2 .* (eps(1,2) + grad2phi1) - grad21phi1 .* (eps(2,1) + grad1phi2))...
                   + grad21phi2 .* norm + (eps(2,2) + grad2phi2) .* partialXnorm2);
    
    %%ggrad_1dir_22 -- page 10
    ggrad_1dir_22 = 4 * (grad22phi1 .* norm + (eps(1,2) + grad2phi1) .* partialYnorm2) - ...
                    ggamma * (2 * grad22phi1 .* determinant + 2 * (eps(1,2) + grad2phi1) .* ...
                    (grad12phi1 .* (eps(2,2) + grad2phi2) + grad22phi2 .* (eps(1,1) + grad1phi1) ...
                    - grad12phi2 .* (eps(1,2) + grad2phi1) - grad22phi1 .* (eps(2,1) + grad1phi2)) ...
                    - grad12phi2 .* norm + (eps(2,1) + grad1phi1) .* partialYnorm2);
    
    %%ggrad_2dir_11 -- page 17
    ggrad_2dir_11 = 4 * (grad11phi2 .* norm + (eps(2,1) + grad1phi2) .* partialXnorm2) -...
                    ggamma * (2 * grad11phi2 .* determinant + 2 * (eps(2,1) + grad1phi2) .* ...
                    (grad11phi1.* (eps(2,2) + grad2phi2) + grad21phi2 .* (eps(1,1) + grad1phi1) ...
                    - grad11phi2 .* (eps(1,2) + grad2phi1) - grad21phi1 .* (eps(2,1) + grad1phi2))...
                    - grad21phi1 .* norm - (eps(1,2) + grad2phi1) .* partialXnorm2);
    
    %%ggrad_2dir_22 -- page 16
    ggrad_2dir_22 = 4 * (grad22phi2 .* norm + (eps(2,2) + grad2phi2) .* partialYnorm2) - ...
                    ggamma * (2 * grad22phi2 .* determinant + 2 * (eps(2,2) + grad2phi2) .* ...
                    (grad12phi1 .* (eps(2,2) + grad2phi2) + grad22phi2 .* (eps(1,1) + grad1phi1) ...
                    - grad12phi2 .* (eps(1,2) + grad2phi1) - grad22phi1 .* (eps(2,1) + grad1phi2)) ...
                    + grad12phi1 .* norm + (eps(1,1) + grad1phi1) .* partialYnorm2);

    %%the functional of J_gamma at phi_k
    updatePhi1 = ggrad_1dir_11 + ggrad_1dir_22;
    updatePhi2 = ggrad_2dir_11 + ggrad_2dir_22;

    update = [-updatePhi1 -updatePhi2];
end

%% inner product function g
function innerProduct = inner(nodes_value,update, alpha, num_div, eps, ggamma)
    phi_k1 = nodes_value - alpha .* update; 
    phi_k1 = process_value(phi_k1);
    update_k1 = functionalJ(phi_k1, num_div, eps, ggamma);
    for i = 1 : (num_div+1)^2
        innerPro(i) = update(i,1) * update_k1(i,1) + update(i,2) * update_k1(i,2);
    end
    innerPro = (1/num_div)^2 .* innerPro;
    innerProduct = sum(innerPro,"all"); 
end





