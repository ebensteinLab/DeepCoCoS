pathIm=fullfile(pwd,"Data","RawImages",'cropGY_full_All-3.tif');
avplot=1;
labelhloc={'left','left','right'};
labelvloc={'bottom','top','middle'};
alpha=[.25,.25,.5];
info=imfinfo(pathIm);
stackFrames=numel(info);
ch_num=6; %all, R, AF594, B, Cy3, G all with 5band
RPAlabels={'RPA=175°','RPA=176°','RPA=177°','RPA=178°','RPA=179°','RPA=180°'};
stackIm=zeros(info(1).Height,info(1).Width,stackFrames);
for i=1:stackFrames
    stackIm(:,:,i)=imread(pathIm,i,'Info',info);
end
hyperstackIm=reshape(stackIm,size(stackIm,1),size(stackIm,2),ch_num,stackFrames/ch_num);
no_dispI=hyperstackIm(:,:,6,size(hyperstackIm,4));
[pks,locsY,w,p]=findpeaks(sum(no_dispI,2),"NPeaks",3,"SortStr","descend");
locsY(3)=locsY(3)+1;

adjLocsX_noDisp=fitPeaks(no_dispI,locsY);

areaColor=[0.4660 0.6740 0.1880
           0.4660 0.6740 0.1880
           0.9290 0.6940 0.1250];	

fig=figure(1);
clf
rpas=linspace(6,1,6);
t=tiledlayout(length(rpas),2,"TileSpacing","compact","Padding","compact");
displocsY=zeros(3,length(rpas));
Ycrop=10:(size(hyperstackIm,1)-1);
Ydisplace=[-2,-1,-1,-1,0,0];
for rpai=1:length(rpas)
    dispIm=hyperstackIm(:,:,6,rpas(rpai));
    nexttile
    imagesc(dispIm(Ycrop+Ydisplace(rpai),20:42),[130 360])

    colormap("gray")

grid off;
set(gca,'TickLength',[0 0],...
    'XTick',[], ...
    'YTick',[])
ylabel(RPAlabels(rpas(rpai)),...
    'FontSize',20,... 
    'FontWeight','Bold')
    

[pks,locsY,w,p]=findpeaks(sum(dispIm,2),"NPeaks",3,"SortStr","descend","WidthReference","halfheight");
displocsY(:,rpai)=locsY;
locsY(2)=locsY(2);%-1;
locsY(1)=locsY(1);

locsY(3)=locsY(3)+1;
y=repmat(-1:1,3,1)+locsY;

adjLocsX_disp=fitPeaks(dispIm,locsY);


hold on
for yi=1:3

end
dispProfiles=zeros(3,size(dispIm,2));
xProfiles=repmat(1:size(dispProfiles,2),3,1)-adjLocsX_noDisp;
nexttile
for yi=1:length(locsY)
dispProfiles(yi,:)=mean(dispIm(y(yi,:),:),1);
if ~avplot
a(yi)=area(xProfiles(yi,:),(dispProfiles(yi,:)-min(dispProfiles(yi,:))),'facealpha',alpha(yi),'EdgeColor',areaColor(yi,:),'EdgeAlpha',.5,'FaceColor',areaColor(yi,:));
hold on
xline(adjLocsX_disp(yi)-adjLocsX_noDisp(yi),':',num2str(adjLocsX_disp(yi)-adjLocsX_noDisp(yi),'%.1f'),'linewidth',2,'color',areaColor(yi,:)*0.9,'LabelVerticalAlignment',labelvloc{yi},'fontsize',16,'fontweight','bold','LabelHorizontalAlignment',labelhloc{yi})
end
end
if ~avplot
    if rpai==1
legend([a(1) a(3)],{'Cy3','AF594'},'Location','northwest')
    end
xlim([-17 3])

set(gca,'FontSize',20,'FontWeight','bold',...
    'YAxisLocation','right')
end
if avplot
    cy3profile=mat2gray(mean([mat2gray(dispProfiles(1,:));mat2gray(interp1(xProfiles(2,:),dispProfiles(2,:),xProfiles(1,:),'linear','extrap'))],1));
    area(xProfiles(1,:),cy3profile,'facealpha',.35,'EdgeColor','none','FaceColor',areaColor(1,:));
    hold on
    area(xProfiles(3,:),mat2gray(dispProfiles(3,:)),'facealpha',.35,'EdgeColor','none','FaceColor',areaColor(3,:));
    xlim([-20 5])
    set(gca,'FontSize',20,'FontWeight','bold',...
        'YAxisLocation','right')
    maxCy3=adjLocsX_disp(1)-adjLocsX_noDisp(1);
    xline(maxCy3,':','linewidth',3,'color',areaColor(1,:).*0.9)
    maxAF594=adjLocsX_disp(3)-adjLocsX_noDisp(3);
    
    if (rpai==1)
    xline(maxAF594,':','linewidth',3,'color',areaColor(3,:).*0.9);%,'LabelVerticalAlignment','top','fontsize',16,'fontweight','bold','LabelHorizontalAlignment','left','LabelOrientation','horizontal');
    text(maxCy3-.5,0.85,['\Delta=',num2str(maxAF594-maxCy3,'%.1f'),' '],'fontsize',18,'HorizontalAlignment','right')

     legend(gca,{'Cy3','AF594'},'Location','northwest')

    elseif (rpai==2)
    xline(maxAF594,':','linewidth',3,'color',areaColor(3,:).*0.9);%,'LabelVerticalAlignment','top','fontsize',16,'fontweight','bold','LabelHorizontalAlignment','left','LabelOrientation','horizontal');
    text(maxCy3-.5,0.85,['\Delta=',num2str(maxAF594-maxCy3,'%.1f'),' '],'fontsize',18,'HorizontalAlignment','right')

    else
    xline(maxAF594,':','linewidth',3,'color',areaColor(3,:).*0.9);%,'LabelVerticalAlignment','top','fontsize',16,'fontweight','bold','LabelHorizontalAlignment','right','LabelOrientation','horizontal');
    text(maxAF594+.5,0.85,['\Delta=',num2str(maxAF594-maxCy3,'%.1f')],'fontsize',18,'HorizontalAlignment','left')
    end
   xlim([-17 3]);
   xl=xlim;

    set(gca,'FontSize',20,'FontWeight','bold',...
    'YAxisLocation','right')
end
end


%%
function adjLocsX=fitPeaks(Im,locsY)
y=repmat(-1:1,3,1)+locsY;
locsX=zeros(size(locsY));
adjLocsX=zeros(size(locsY));
adj_pk=zeros(size(locsY));
adj_w=zeros(size(locsY));
adj_skew=zeros(size(locsY));
for yi=1:3

 peak_profile= mean(Im(y(yi,:),:),1);
 x=[1:length(peak_profile)]';

adj_peak_profile=peak_profile-max(peak_profile(1:20));    
adj_peak_profile(adj_peak_profile<0)=0;
adjLocsX(yi)=sum(x.*adj_peak_profile')/sum(adj_peak_profile');%a(2);

end
end