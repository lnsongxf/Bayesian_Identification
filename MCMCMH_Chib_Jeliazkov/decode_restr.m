function constr = decode_restr(integer, n, m)

constr = ones(m, n);
for i = 1:m
    for j = 1:n
        if i == j
            continue
        end
        if mod(integer, 2) == 1
            constr(i, j) = 0;
        end
        integer = (integer - mod(integer, 2))/2;
    end
end