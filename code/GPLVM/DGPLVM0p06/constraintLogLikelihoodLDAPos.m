function ll = constraintLogLikelihoodLDAPos(model,X);

% CONSTRAINTLOGLIKELIHOODLDAPOS Returns loglikelihood for LDAPos constraint
%
%	Description:
%
%	LL = CONSTRAINTLOGLIKELIHOODLDAPOS(MODEL, X) Returns loglikelihood
%	for LDAPos constraint
%	 Returns:
%	  LL - Returns loglikelihood
%	 Arguments:
%	  MODEL - fgplvm model
%	  X - Latent locations
%	
%
%	See also
%	CONSTRAINTLOGLIKELIHOOD


%	Copyright (c) 2009 Carl Henrik Ek
% 	constraintLogLikelihoodLDAPos.m SVN version 817
% 	last update 2010-05-31T19:18:36.000000Z

ll = -model.lambda*trace(model.A);

return