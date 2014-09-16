function model = constraintExpandParam(model,X)

% CONSTRAINTEXPANDPARAM Expands a constraint model
%
%	Description:
%
%	RETURN = CONSTRAINTEXPANDPARAM(MODEL, X) Returns updated model
%	 Returns:
%	  RETURN - constraint model
%	 Arguments:
%	  MODEL - constraint model
%	  X - Latent locations
%	
%
%	See also
%	


%	Copyright (c) 2009 Carl Henrik Ek
% 	constraintExpandParam.m SVN version 295
% 	last update 2009-03-08T08:54:09.000000Z

fhandle = str2func(['constraintExpandParam',model.type]);
model = fhandle(model,X);

return;