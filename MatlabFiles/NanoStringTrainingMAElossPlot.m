folderName=fullfile(pwd,"TrainingHistory");


matFiles = dir(fullfile(folderName,'*.mat'));
history123 = cell(1, length(matFiles)/2);
history4 = cell(1, length(matFiles)/2);
fovNum=zeros(1,length(matFiles)/2);
for k = 1:length(matFiles)/2

    history123{k} = load(fullfile(folderName,matFiles(2*k-1).name));
    ind1=strfind(matFiles(2*k-1).name,'subset');
    ind2=strfind(matFiles(2*k-1).name,'123c');
    fovNum(k)=sscanf(matFiles(2*k-1).name(ind1+length('subset'):ind2-1),'%d',1);
    history4{k}=load(fullfile(folderName,matFiles(2*k).name));
end

%%
figure (1)
clf
t=tiledlayout("flow",'TileSpacing','tight','Padding','tight');
[~,sorting_ind]=sort(fovNum,'ascend');
historyMat123=zeros(6,200,length(fovNum));
historyMat4=zeros(6,200,length(fovNum));
testMat123=zeros(2,1,length(fovNum));
testMat4=zeros(2,1,length(fovNum));
for i=1:length(fovNum)
historyCell123=struct2cell(history123{sorting_ind(i)});
historyMat123(:,:,i)=cell2mat(historyCell123(1:6));
testMat123(:,:,i)=cell2mat(historyCell123(7:end));

historyCell4=struct2cell(history4{sorting_ind(i)});
historyMat4(:,:,i)=cell2mat(historyCell4(1:6));
testMat4(:,:,i)=cell2mat(historyCell4(7:end));
end

epoch=1:size(historyMat123,2);
ylimsmae=[5 max([historyMat123([1,4],:,length(fovNum));historyMat4([1,4],:,length(fovNum))],[],"all")];
ylimsmse=[0 max([historyMat123([3,6],:,length(fovNum));historyMat4([3,6],:,length(fovNum))],[],"all")];

C=colororder('default');


for i=1:size(historyMat123,3)-1
    nexttile
    plot(epoch,[historyMat123([1,4],:,i);historyMat4([1,4],:,i)],'LineWidth',2,'Marker','none','LineStyle','-');
    if mod(i,4)==1

        ylabel('Loss (MAE) value')
    end
    hold on
    xline(find(historyMat123(4,:,i)==min(historyMat123(4,:,i)),1,'first'),'LineWidth',2,'Color',C(2,:))
    xline(find(historyMat4(4,:,i)==min(historyMat4(4,:,i)),1,'first'),'LineWidth',2,'Color',C(4,:))
    yline(min(historyMat123(4,:,i)),':','LineWidth',2,'Color',C(2,:),'Label',[num2str(min(historyMat123(4,:,i)),'%.1f'),10],'LabelHorizontalAlignment','center','LabelVerticalAlignment','top','LabelOrientation','horizontal','FontSize',16,'FontWeight','bold')
    yline(min(historyMat4(4,:,i)),':','LineWidth',2,'Color',C(4,:),'Label',[num2str(min(historyMat4(4,:,i)),'%.1f')],'LabelHorizontalAlignment','center','LabelVerticalAlignment','bottom','LabelOrientation','horizontal','FontSize',16,'FontWeight','bold')
    
    

    ylim(ylimsmae)
    title ([int2str(fovNum(sorting_ind(i))),' FOV',10,'RYB: ',num2str(testMat123(1,1,i),'%.2f'),'    G: ',num2str(testMat4(1,1,i),'%.2f')],'FontSize',20,'FontWeight','bold')
    if i>8
        xlabel('Epoch')
        set(gca,'FontSize',20,'FontWeight','bold','LineWidth',1)
    else
        set(gca,'FontSize',20,'FontWeight','bold','XTickLabel',[],'LineWidth',1)

    end
    if i==1
        
        legend('RYB Training','RYB Validation','G Training','G Validation','fontsize',12,'location','north')
    end

end

title (t,[" RYB mean pixel intensity: 162\pm67","G mean pixel intensity: 120\pm54"],'FontSize',20,'FontWeight','bold')
%

figure(2)

t2=tiledlayout("flow",'TileSpacing','tight','Padding','tight');

C=colororder('default');


for i=1:size(historyMat123,3)-1
    nexttile
    plot(epoch,[historyMat123([3,6],:,i);historyMat4([3,6],:,i)],'LineWidth',2,'Marker','none','LineStyle','-');
    if mod(i,4)==1

        ylabel('MSE value')
    end
    hold on
    xline(find(historyMat123(6,:,i)==min(historyMat123(6,:,i)),1,'first'),'LineWidth',2,'Color',C(2,:))
    xline(find(historyMat4(6,:,i)==min(historyMat4(6,:,i)),1,'first'),'LineWidth',2,'Color',C(4,:))
    yline(min(historyMat123(6,:,i)),':','LineWidth',2,'Color',C(2,:),'Label',[num2str(min(historyMat123(6,:,i)),'%.1f')],'LabelHorizontalAlignment','right','LabelVerticalAlignment','top','LabelOrientation','horizontal','FontSize',16,'FontWeight','bold')
    yline(min(historyMat4(6,:,i)),':','LineWidth',2,'Color',C(4,:),'Label',[num2str(min(historyMat4(6,:,i)),'%.1f')],'LabelHorizontalAlignment','right','LabelVerticalAlignment','bottom','LabelOrientation','horizontal','FontSize',16,'FontWeight','bold')
    
    

    ylim(ylimsmse)
    title ([int2str(fovNum(sorting_ind(i))),' FOV',10,'RYB: ',num2str(testMat123(2,1,i),'%.2f'),'    G: ',num2str(testMat4(2,1,i),'%.2f')],'FontSize',20,'FontWeight','bold')
    if i>8
        xlabel('Epoch')
        set(gca,'FontSize',20,'FontWeight','bold','LineWidth',1)
    else
        set(gca,'FontSize',20,'FontWeight','bold','XTickLabel',[],'LineWidth',1)

    end
    if i==1
        
        legend('RYB Training','RYB Validation','G Training','G Validation','fontsize',12,'location','north')
    end

end

title (t2,[" RYB mean pixel intensity: 162\pm67","G mean pixel intensity: 120\pm54"],'FontSize',20,'FontWeight','bold')
%
%%
figure(3)
clf
%4,ceil(length(history4)/2)
t3=tiledlayout(2,1,'TileSpacing','tight','Padding','tight');
C=colororder('default');

i=length(fovNum);
nexttile
plot(epoch,[historyMat123([1,4],:,i);historyMat4([1,4],:,i)],'LineWidth',2,'Marker','none','LineStyle','-');
ylabel('Loss (MAE) value')
hold on
xline(find(historyMat123(4,:,i)==min(historyMat123(4,:,i)),1,'first'),'LineWidth',2,'Color',C(2,:))
xline(find(historyMat4(4,:,i)==min(historyMat4(4,:,i)),1,'first'),'LineWidth',2,'Color',C(4,:))
yline(min(historyMat123(4,:,i)),':','LineWidth',2,'Color',C(2,:),'Label',[num2str(min(historyMat123(4,:,i)),'%.1f')],'LabelHorizontalAlignment','center','LabelVerticalAlignment','top','LabelOrientation','horizontal','FontSize',16,'FontWeight','bold')
yline(min(historyMat4(4,:,i)),':','LineWidth',2,'Color',C(4,:),'Label',[num2str(min(historyMat4(4,:,i)),'%.1f')],'LabelHorizontalAlignment','center','LabelVerticalAlignment','bottom','LabelOrientation','horizontal','FontSize',16,'FontWeight','bold')
ylim(ylimsmae)
title (['Trained on ',int2str(fovNum(sorting_ind(i))),' FOV',10,'Test scores - RYB: ',num2str(testMat123(1,1,i),'%.2f'),'    G: ',num2str(testMat4(1,1,i),'%.2f')],'FontSize',20,'FontWeight','bold')
set(gca,'FontSize',20,'FontWeight','bold','XTickLabel',[],'LineWidth',1)
legend('RYB Training','RYB Validation','G Training','G Validation','fontsize',16,'location','northeast')

nexttile
plot(epoch,[historyMat123([3,6],:,i);historyMat4([3,6],:,i)],'LineWidth',2,'Marker','none','LineStyle','-');
ylabel('MSE value')
hold on
xline(find(historyMat123(6,:,i)==min(historyMat123(6,:,i)),1,'first'),'LineWidth',2,'Color',C(2,:))
xline(find(historyMat4(6,:,i)==min(historyMat4(6,:,i)),1,'first'),'LineWidth',2,'Color',C(4,:))
yline(min(historyMat123(6,:,i)),':','LineWidth',2,'Color',C(2,:),'Label',[num2str(min(historyMat123(6,:,i)),'%.1f'),10],'LabelHorizontalAlignment','right','LabelVerticalAlignment','top','LabelOrientation','horizontal','FontSize',16,'FontWeight','bold')
yline(min(historyMat4(6,:,i)),':','LineWidth',2,'Color',C(4,:),'Label',[num2str(min(historyMat4(6,:,i)),'%.1f'),10],'LabelHorizontalAlignment','right','LabelVerticalAlignment','top','LabelOrientation','horizontal','FontSize',16,'FontWeight','bold')
ylim(ylimsmse)
title (['Test scores - RYB: ',num2str(testMat123(2,1,i),'%.2f'),'    G: ',num2str(testMat4(2,1,i),'%.2f')],'FontSize',20,'FontWeight','bold')
xlabel('Epoch')
set(gca,'FontSize',20,'FontWeight','bold','LineWidth',1)

title (t3,[" RYB mean pixel intensity: 162\pm67","G mean pixel intensity: 120\pm54"],'FontSize',20,'FontWeight','bold')

