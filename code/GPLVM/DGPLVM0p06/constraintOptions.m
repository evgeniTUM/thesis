function options = constraintOptions(type)

% CONSTRAINTOPTIONS Return default options for latent constraint.
%
%	Description:
%
%	OPTIONS = CONSTRAINTOPTIONS(TYPE) Returns the defualt options for a
%	latent constraint
%	 Returns:
%	  OPTIONS - structure containing defualt options
%	 Arguments:
%	  TYPE - Type of constraint as string
%	
%
%	See also
%	FGPLVMADDCONSTRAINT, SGPLVMADDCONSTRAINT


%	Copyright (c) 2009 Carl Henrik Ek
% 	constraintOptions.m SVN version 815
% 	last update 2010-05-31T19:18:36.000000Z

switch type
 case 'LDA'
  options.type = type;
  options.lambda = 1;
  options.unknown = -1;
  options.class = [];
  options.dim = [];
  options.reg = 1;
 case 'LDAPos'
  options.type = type;
  options.lambda = 1;
  options.unknown = -1;
  options.class = [];
  options.dim = [];
  options.reg = 1;
 case 'LDANeg'
  options.type = type;
  options.lambda = 1;
  options.unknown = -1;
  options.class = [];
  options.dim = [];
  options.reg = 1;
 case 'LLE'
  options.type = type;
  options.lambda = 1;
  options.nn = 7;
  options.tol = 1e-3;
 case 'CorrespondenceSequenceLDA'
  options.type = type;
  options.lambda = 1;
  options.class = [];
  options.dim = [];
 otherwise
  error('Unknown constraint');
end


return