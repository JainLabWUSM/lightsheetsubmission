input = getDirectory("input folder?: ");
output = getDirectory("Output folder?: ");
list = getFileList(input);
setBatchMode(true);

	for(i = 0; i < list.length; i++) {
		open(input+list[i]);
		run("Skeletonize (2D/3D)");
		saveAs("Tiff", output + i);
		run("Close");
	}
