function out = proximal_operator_SCAD(paras,  alpha,coff)
out = zeros(size(paras));
for ii = 1:length(out) - 1
    if abs(paras(ii)) < 2*alpha
        out(ii) = sign(paras(ii)).* max(0, abs(paras(ii)) -  alpha);
    elseif abs(paras(ii))<coff*alpha
        alphatmp = ((coff-1)*paras(ii) - sign(paras(ii))*coff*alpha )/(coff - 2) ;
        out(ii) = alphatmp ;
    else
        out(ii) = paras(ii);
    end
end
out(end) = paras(end);
end