function ll = constraintLogLikelihood(model,X)

% CONSTRAINTLOGLIKELIHOOD Wrapper for Constraint loglikelihood
%
%	Description:
%
%	LL = CONSTRAINTLOGLIKELIHOOD(MODEL, X) Returns loglikelihood for
%	constraint
%	 Returns:
%	  LL - Returns loglikelihood
%	 Arguments:
%	  MODEL - fgplvm model
%	  X - Latent locations
%	
%
%	See also
%	


%	Copyright (c) 2009 Carl Henrik Ek
% 	constraintLogLikelihood.m SVN version 817
% 	last update 2010-05-31T19:18:36.000000Z

fhandle = str2func(['constraintLogLikelihood' model.type]);
ll = fhandle(model,X);

return