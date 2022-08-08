//#@ File (style="directory") imageFolder
//#@ File (style="open") beadMaskFile
#@ File (style="open") gtFile
//#@ File (style="open") predFile
//#@ File (style="open") templateFile
//

setBatchMode(true);
close("*");
run("Clear Results");
print(gtFile);
open(gtFile);
mask_name=getInfo("selection.name");
getDimensions(width, height, channels, slices, frames);
if (slices>1&&frames==1){
run("Re-order Hyperstack ...", "channels=[Frames (t)] slices=[Channels (c)] frames=[Slices (z)]");
}else{
	run("Re-order Hyperstack ...", "channels=[Slices (z)] slices=[Channels (c)] frames=[Frames (t)]");
}
run("Z Project...", "projection=[Sum Slices] all");


run("Set Measurements...", "area mean standard min redirect=None decimal=3");
run("Measure Stack...", "frames order=czt(default)");
Rows=nResults;
framesToDelete=newArray;

if (frames==1){
run("Re-order Hyperstack ...", "channels=[Channels (c)] slices=[Frames (t)] frames=[Slices (z)]");
}


s=0;
for (row = 0; row < Rows; row++) {
	if(getResult("Mean", row)<500||getResult("Mean", row)>700||getResult("StdDev", row)<50||getResult("StdDev", row)>1000||getResult("Max", row)<3000){
	framesToDelete[s] = getResult("Slice", row);
	s++;
	}
}

framesToDelete=Array.reverse(framesToDelete);
Array.print(framesToDelete);

/////////////////////////////////////////////////
for (i = 0; i < framesToDelete.length; i++) {
Stack.setPosition(1,1, framesToDelete[i]);

//waitForUser;
run("Delete Slice", "delete=frame");
}
save(File.getDirectory(gtFile)+"filtered_SUM_"+File.getName(gtFile));
//close();
//////////////////////////////////////////////
//open(predFile);

//pred_name=getInfo("selection.name");
//getDimensions(width, height, channels, slices, frames);
//if (frames==1&&slices>1){
//run("Re-order Hyperstack ...", "channels=[Channels (c)] slices=[Frames (t)] frames=[Slices (z)]");
//}
//for (i = 0; i < framesToDelete.length; i++) {
////setSlice(framesToDelete[i]);
//Stack.setPosition(1, 1, framesToDelete[i]);
//run("Delete Slice", "delete=frame");
//}
//save(File.getDirectory(predFile)+"filtered_"+File.getName(predFile));
//close();
///////////////////////////////////////////////
open(gtFile);

gt_name=getInfo("selection.name");
getDimensions(width, height, channels, slices, frames);
if (frames==1&&slices>1){
run("Re-order Hyperstack ...", "channels=[Channels (c)] slices=[Frames (t)] frames=[Slices (z)]");
}
for (i = 0; i < framesToDelete.length; i++) {
//setSlice(framesToDelete[i]);
Stack.setPosition(1, 1, framesToDelete[i]);
run("Delete Slice", "delete=frame");
}
save(File.getDirectory(gtFile)+"filtered_"+File.getName(gtFile));
close();
//open(templateFile);
//run("Template Matching Image");
/////////////////////////////////////////////
//////////////////////////////////////////////


setBatchMode(false);