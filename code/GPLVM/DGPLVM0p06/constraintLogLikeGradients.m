function gX = constraintLogLikeGradients(model)

% CONSTRAINTLOGLIKEGRADIENTS Wrapper for constraint loglike gradients
%
%	Description:
%
%	OPTIONS = CONSTRAINTLOGLIKEGRADIENTS(MODEL) Returns loglikelihood
%	for LDAPos constraint
%	 Returns:
%	  OPTIONS - Returns loglike gradients
%	 Arguments:
%	  MODEL - fgplvm model
%	
%
%	See also
%	CONSTRAINTLOGLIKELLIHOOD


%	Copyright (c) 2009 Carl Henrik Ek
% 	constraintLogLikeGradients.m SVN version 295
% 	last update 2009-03-08T08:54:13.000000Z

fhandle = str2func(['constraintLogLikeGradients',model.type]);
gX = fhandle(model);

return