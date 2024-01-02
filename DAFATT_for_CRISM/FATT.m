function     [kf, NorRMSE, model]=FATT(data,targetlibrary,varargin)  

% Using Factor Analysis and Target Transformation(FATT) to find the endmembers
%  Reference: "Identification and refinement of martian surface mineralogy
%  using factor analysis and target transformation of near-infrared
%  spectroscopic data". Nancy H. Thomas and Joshua L. Bandfield, Icarus, 2017. 
%% --------------- Description -------------------------------------------
%  output: the blue spectra are the modeled, the orange spectra are pure endmembers 

%%  ===== Required inputs =============
%  data - [L(channels) x N(pixel number)] mixing matrix
% targetlibrary-[L(channels) x P(target spectra number)] 
% targetlibraryName-the mineral name of each spectrunm in target library 
% wavelength-L*1, unit: micrometer 
%%  ===== Outputs =============
% kf-The eigenvector number 
% NorRMSE-RMSE between the normalized library spectra and modeled spectra 

%%  ===== Optional inputs =============================
%
%  'EigNumDM' - eigenvector number determination method 
%                          including 'FATTPaper','SpectralInfo','Hysime'.
%                          suggest 'Hysime' 
%                          Default: 'FATTPaper'   

%   Copyright: Honglei Lin (linhonglei9@gmail.com)
%            & Jesse Tarnas (jesse_tarnas@brown.edu)
%  Honglei Lin, J,D.Tarnas, J. F. Mustard, Xia Zhang et al. Dynamic Aperture Factor Analysis/Target Transformation (DAFA/TT)
%  for Serpentine and Mg-Carbonate Mapping on Mars with CRISM Near-Infrared Data. Icarus, 2020.

eignum='FATTPaper';%default 

if (nargin-length(varargin)) ~= 4
    error('Wrong number of required parameters');
end
% data matrix size
[LM,N] = size(data);
% target library size
[L,P] = size(targetlibrary);
if (LM ~= L)
    error('Data matrix and target library are inconsistent');
end

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'EIGNUMDM'
                eignum = varargin{i+1};
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);     
        end
    end
end


%Factor analysis
    Normaldata=data-repmat(mean(data,2),[1,N]);
    C=Normaldata*Normaldata'/(N);
    [eigenvectors,eigenvalues]=eig(C);
    eigva=diag(eigenvalues);
    pp=L:-1:1;
    eigva=eigva(pp);
    eigenvectors=eigenvectors(:,pp);
if strcmp(eignum,'Hysime')
    noise_type = 'additive';verbose='off';
    [w Rn] = estNoise(data,noise_type,verbose);
    [kf Ek]=hysime(data,w,Rn,verbose);
else   
   
    if strcmp(eignum,'SpectralInfo') 
        Info=sum(eigva);
        cum=abs(cumsum(eigva)/Info-0.9995);
        kf=find(cum==min(cum));  
        Ek=eigenvectors(:,1:kf);
    else
        if strcmp(eignum,'FATTPaper') 
            kf=10;%defaut eigenvector number 
            Ek=eigenvectors(:,1:kf);
        end
    end
end


% Target Transform 
targetlibraryNor=targetlibrary./repmat(sum(targetlibrary),[L,1]);
for i=1:size(targetlibrary,2)
   X_hat_tv_i=hyperUcls(targetlibrary(:,i),Ek);
   model(:,i)=Ek*X_hat_tv_i;    
   model1(:,i)=model(:,i)./repmat(sum(model(:,i)),[L,1]);
   %RMSE(i)=sqrt(sum((model(:,i)-TargetLibraryRef(:,i)).^2)/nband);
   NorRMSE(i)=sqrt(sum((model1(:,i)-targetlibraryNor(:,i)).^2)/L);
end
