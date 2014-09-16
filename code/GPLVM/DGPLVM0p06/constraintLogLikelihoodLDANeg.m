function ll = constraintLogLikelihoodLDANeg(model,X)

% CONSTRAINTLOGLIKELIHOODLDANEG Constraint loglikelihood LDA Neg model
%
%	Description:
%
%	LL = CONSTRAINTLOGLIKELIHOODLDANEG(MODEL, X) Returns loglikelihood
%	for constraint
%	 Returns:
%	  LL - Returns loglikelihood
%	 Arguments:
%	  MODEL - fgplvm or sgplvm model
%	  X - Latent locations
%	
%
%	See also
%	


%	Copyright (c) 2009 Carl Henrik Ek
% 	constraintLogLikelihoodLDANeg.m SVN version 911
% 	last update 2010-08-23T05:54:38.000000Z


ll = -model.lambda*trace(model.A);

return;