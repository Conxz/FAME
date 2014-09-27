FAME
====

FSL-Feat Automatic Motion Extraction from Feat outputs.

Installation
------------

	pip install git+https://github.com/Conxz/FAME.git

Usage
-----

	usage: calc_featheadmotion [-h] -sd sess-dir -sf sess-file -exp exp-name -rlf
                           runlist-file -feat feat-dir -thr thr thr -o output
                           [--log log-file] [-v]

	A cmd-line for getting head motion.

	optional arguments:
  		-h, --help         show this help message and exit
  		-sd sess-dir       The sessdir.
 		-sf sess-file      The sess list file.
  		-exp exp-name      The exp name.
  		-rlf runlist-file  The run list file.
  		-feat feat-dir     The feat dir.
  		-thr thr thr       The thresh for motion, absthr relthr.
  		-o output          The output file.
  		--log log-file     log name for the processing.
  		-v                 show program version number and exit

Example
-------
	coming soon...
