function ll = constraintLogLikelihoodLDA(model,X)

% CONSTRAINTLOGLIKELIHOODLDA Constraint loglikelihood for LDA model
%
%	Description:
%
%	LL = CONSTRAINTLOGLIKELIHOODLDA(MODEL, X) Returns loglikelihood for
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
% 	constraintLogLikelihoodLDA.m SVN version 817
% 	last update 2010-05-31T19:18:36.000000Z


ll = -model.lambda*trace(model.A);

return;