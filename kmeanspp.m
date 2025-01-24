function [label, centroid, energy] =  kmeanspp(X, k, varargin)
% X: d x n data matrix
% k: number of seeds
% 
% reference: k-means++: the advantages of careful seeding. David Arthur and Sergei Vassilvitskii
% 
% Written by Michael Chen (sth4nth@gmail.com).
% Modified by Yeqing Li (yeqing.li@mavs.uta.edu).
% Corrected by Feiping Nie, 7/26/2014

pnames = {'maxiter'  'replicates'};
dflts =  {     []        []      };
[eid,errmsg,maxit,reps] = getargs(pnames, dflts, varargin{:});
if ~isempty(eid)
    error(sprintf('kmeanspp:%s',eid),errmsg);
end

% Assume one replicate
if isempty(reps) 
    reps = 1;
end

bestenergy = inf;

for t=1:reps,

    m = seeds(X,k);
    [label, energy] = kmeans(X, m);
    % compute centroid of each cluster
    n = length(label);
    ind = sparse(label,1:n,1,k,n,n);
    centroid = (spdiags(1./sum(ind,2),0,k,k)*ind)*X';

    if t == 1 || energy < bestenergy,
        bestlabel = label;
        bestcenters = centroid;
        bestenergy = energy;
    end
end
label = bestlabel';
centroid = bestcenters;

%------------------------------------------------------------
function m = seeds(X, k)
[d,n] = size(X);
if issparse(X),
    m = sparse(d,k);
else
    m = zeros(d,k);
end
v = inf(1,n);
m(:,1) = X(:,ceil(n*rand));
for i = 2:k
    Y = bsxfun(@minus,X,m(:,i-1));
    %v = cumsum(min(v,dot(Y,Y,1)));
    v = min(v,dot(Y,Y,1));
    v1 = cumsum(v);
    m(:,i) = X(:,find(rand < v1/v1(end),1));
end

function [label, energy] = kmeans(X, m)
n = size(X,2);
last = 0;
[~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
while any(label ~= last)
    [u,~,label] = unique(label);   % remove empty clusters
    label = label(:)';
    k = length(u);
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
    m = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute m of each cluster
    last = label;
    [value,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
end
[~,~,label] = unique(label);   % remove empty clusters
energy = -2*sum(value)+dot(X(:),X(:)); 

%------------------------------------------------------------
function [eid,emsg,varargout]=getargs(pnames,dflts,varargin)
%GETARGS Process parameter name/value pairs 
%   [EID,EMSG,A,B,...]=GETARGS(PNAMES,DFLTS,'NAME1',VAL1,'NAME2',VAL2,...)
%   accepts a cell array PNAMES of valid parameter names, a cell array
%   DFLTS of default values for the parameters named in PNAMES, and
%   additional parameter name/value pairs.  Returns parameter values A,B,...
%   in the same order as the names in PNAMES.  Outputs corresponding to
%   entries in PNAMES that are not specified in the name/value pairs are
%   set to the corresponding value from DFLTS.  If nargout is equal to
%   length(PNAMES)+1, then unrecognized name/value pairs are an error.  If
%   nargout is equal to length(PNAMES)+2, then all unrecognized name/value
%   pairs are returned in a single cell array following any other outputs.
%
%   EID and EMSG are empty if the arguments are valid.  If an error occurs,
%   EMSG is the text of an error message and EID is the final component
%   of an error message id.  GETARGS does not actually throw any errors,
%   but rather returns EID and EMSG so that the caller may throw the error.
%   Outputs will be partially processed after an error occurs.
%
%   This utility can be used for processing name/value pair arguments.
%
%   Example:
%       pnames = {'color' 'linestyle', 'linewidth'}
%       dflts  = {    'r'         '_'          '1'}
%       varargin = {{'linew' 2 'nonesuch' [1 2 3] 'linestyle' ':'}
%       [eid,emsg,c,ls,lw] = statgetargs(pnames,dflts,varargin{:})    % error
%       [eid,emsg,c,ls,lw,ur] = statgetargs(pnames,dflts,varargin{:}) % ok

% We always create (nparams+2) outputs:
%    one each for emsg and eid
%    nparams varargs for values corresponding to names in pnames
% If they ask for one more (nargout == nparams+3), it's for unrecognized
% names/values

%   Original Copyright 1993-2008 The MathWorks, Inc. 
%   Modified by Deng Cai (dengcai@gmail.com) 2011.11.27




% Initialize some variables
emsg = '';
eid = '';
nparams = length(pnames);
varargout = dflts;
unrecog = {};
nargs = length(varargin);

% Must have name/value pairs
if mod(nargs,2)~=0
    eid = 'WrongNumberArgs';
    emsg = 'Wrong number of arguments.';
else
    % Process name/value pairs
    for j=1:2:nargs
        pname = varargin{j};
        if ~ischar(pname)
            eid = 'BadParamName';
            emsg = 'Parameter name must be text.';
            break;
        end
        i = strcmpi(pname,pnames);
        i = find(i);
        if isempty(i)
            % if they've asked to get back unrecognized names/values, add this
            % one to the list
            if nargout > nparams+2
                unrecog((end+1):(end+2)) = {varargin{j} varargin{j+1}};
                % otherwise, it's an error
            else
                eid = 'BadParamName';
                emsg = sprintf('Invalid parameter name:  %s.',pname);
                break;
            end
        elseif length(i)>1
            eid = 'BadParamName';
            emsg = sprintf('Ambiguous parameter name:  %s.',pname);
            break;
        else
            varargout{i} = varargin{j+1};
        end
    end
end

varargout{nparams+1} = unrecog;
