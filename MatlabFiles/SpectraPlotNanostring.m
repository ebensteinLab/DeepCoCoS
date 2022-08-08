myFolder=fullfile(pwd,"FluorophoresSpectra");
filePattern = fullfile(myFolder, '*.txt'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
S=cell(1,length(theFiles));
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, '%d Now reading %s\n', k,baseFileName);  
  S{k}=load(fullFileName,'-ascii');
end
%%
gt_flag=0;
FluorLabel={'AF 488','Cy 3','AF 594','AF 647'};
wl_lim=[450 750];
Sadj=cellfun(@(x) x(x(:,1)>wl_lim(1)&x(:,1)<wl_lim(2),:),S,'UniformOutput',false);
SpectralOffset=-10;
LaserLines=[488,561,638];
Filters=FilterSpectrum(0,0);
Filters=Filters(:,[2,3,3,4]);
if gt_flag
Filters=[519.5,575,620,693;25,15,14,39];
end
PatchFiltersX=[Filters(1,:)'-Filters(2,:)'./2,Filters(1,:)'-Filters(2,:)'./2,Filters(1,:)'+Filters(2,:)'./2,Filters(1,:)'+Filters(2,:)'./2];
PatchFiltersY=repmat([0,1,1,0],size(Filters,2),1);
maxLambdaFl=cellfun(@(x) max(x(x(:,2)==max(x(:,2)),1)), Sadj,'UniformOutput',true);
[maxLambdaFlSorted,indLambdaFl]=sort(maxLambdaFl);
Ssorted=Sadj(indLambdaFl);
sRGBFl=squeeze(spectrumRGB(maxLambdaFlSorted+SpectralOffset));

sRGBLasers=spectrumRGB(LaserLines);


fig=figure();
clf
hSpectra=axes(fig);
hold on

AxesH = axes('Parent', fig, ...
  'Units', 'normalized', ...
  'Position', [hSpectra.Position(1), 0, hSpectra.Position(3), 1], ...
  'Visible', 'off', ...
  'XLim', [0, 1], ...
  'YLim', [0, 1], ...
  'NextPlot', 'add');
hold on
linkaxes([hSpectra,AxesH],'x')
xlim(hSpectra,wl_lim)
for i=1:2:length(Ssorted)
    if gt_flag
    h_ex(i)=plot(hSpectra,Ssorted{i}(:,1),Ssorted{i}(:,2),'LineStyle',':','color',sRGBFl(i,:),'LineWidth',2);
    end
x_ind_for_color=Ssorted{i+1}(:,1)>=PatchFiltersX(floor(i/2)+1,1)&Ssorted{i+1}(:,1)<=PatchFiltersX(floor(i/2)+1,3);

h_gray(i)=area(hSpectra,Ssorted{i+1}(:,1),Ssorted{i+1}(:,2),'FaceColor',[.5 .5 .5],'FaceAlpha',0.3,'LineStyle','--');
h_em(i)=area(hSpectra,Ssorted{i+1}(x_ind_for_color,1),Ssorted{i+1}(x_ind_for_color,2),'FaceColor',sRGBFl(i,:),'FaceAlpha',0.5,'LineStyle','none');
if i==5&&gt_flag
    x_ind_for_color=Ssorted{(i-2)+1}(:,1)>=PatchFiltersX(floor((i)/2)+1,1)&Ssorted{(i-2)+1}(:,1)<=PatchFiltersX(floor((i)/2)+1,3);
        
area(hSpectra,Ssorted{(i-2)+1}(x_ind_for_color,1),Ssorted{(i-2)+1}(x_ind_for_color,2),'FaceColor',sRGBFl(i-1,:)-[0.65 0.05 0],'FaceAlpha',.5,'LineStyle','none');

end
text(AxesH,maxLambdaFlSorted(i+1),0.95, FluorLabel{floor(i/2)+1},'HorizontalAlignment','center', 'BackgroundColor', 'none','FontSize',24,'Rotation',0);
end
for i=1:length(LaserLines)
    hx(i)=xline(hSpectra,LaserLines(i),'color',sRGBLasers(1,i,:),'linewidth',5);
    text(hSpectra,LaserLines(i)-6,0.5, [num2str(LaserLines(i)),' nm'],'HorizontalAlignment','center', 'BackgroundColor', 'none','FontSize',24,'Rotation',90);

end
for i=1:size(PatchFiltersX,1)

    hp(i)=patch(hSpectra,PatchFiltersX(i,:),PatchFiltersY(i,:),sRGBFl(2*(i-1)+1,:),'faceAlpha',0.05,'linestyle',':');

    if i==2&&~gt_flag
        set(hp(i),'linestyle',':','linewidth',2)
    end
end

set(hSpectra,...
    'FontSize',24,...
    'Box','on',...
    'LineWidth',2,...
    'BoxStyle','full',...
    'YTick',[]);
ylim(hSpectra,[0 1])
xlabel(hSpectra,'Wavelength [nm]');
if gt_flag
hSpectra.Children(11:23)=hSpectra.Children([21,18,14,15,11,23,22,20,19,17,16,13,12]);
hSpectra.Children=hSpectra.Children(circshift(1:numel(hSpectra.Children),19));
else
hSpectra.Children([13,14,15,16])=hSpectra.Children([15,16,13,14]);
hSpectra.Children=hSpectra.Children(circshift(1:numel(hSpectra.Children),14));
end
