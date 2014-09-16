function model = constraintCreate(model,void,options,varargin)

% CONSTRAINTCREATE Creates a constraint model from a options struct
%
%	Description:
%
%	MODEL = CONSTRAINTCREATE(MODEL, VOID, OPTIONS) Creates a constraint
%	model from a options struct
%	 Returns:
%	  MODEL - the model created
%	 Arguments:
%	  MODEL - fgplvm or sgplvm model
%	  VOID - empty for compatibility reasons
%	  OPTIONS - options structure as returned by constraintOptions
%	
%
%	See also
%	CONSTRAINTOPTIONS


%	Copyright (c) 2009, 2010 Carl Henrik Ek
% 	constraintCreate.m SVN version 817
% 	last update 2010-05-31T19:18:36.000000Z

options.q = model.q;
options.N = model.N;

fhandle = str2func(['constraintCreate' options.type]);
if(isfield(model.constraints,'numConstraints')&&model.constraints.numConstraints>0)
  model.constraints.comp{model.constraints.numConstraints+1} = fhandle(options,varargin{:});
else
  model.constraints.comp{1} = fhandle(options,varargin{:});
end

return
