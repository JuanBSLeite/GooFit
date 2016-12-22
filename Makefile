.PHONY:		goofit clean examples

GOOFITDIR = $(shell /bin/pwd)
include Makefile.common

FUNCTORLIST   = $(wildcard src/PDFs/*.cu)
FUNCTORLIST   += $(wildcard include/goofit/PDFs/*.h)

# Handy print for makefile variables (debugging)
print-%  : ; @echo $* = $($*)

ROOTRIPOBJS	= wrkdir/TMinuit.o wrkdir/TRandom.o wrkdir/TRandom3.o 

goofit: $(THRUSTO_B) wrkdir/libRootUtils.so
	@echo "Built GooFit objects" 

# One rule for GooFit objects.
wrkdir/%.o:	src/goofit/%.cc include/goofit/%.h 
	@mkdir -p wrkdir
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c -o $@ $<


# A different rule for user-level objects. Notice ROOT_INCLUDES. 
%.o:	%.cu
	$(CXX) $(INCLUDES) $(ROOT_INCLUDES)  $(CXXFLAGS) -c -o $@ $<

# Still a third rule for the ROOT objects - these have their own Makefile. 
$(ROOTRIPDIR)/%.o:	$(ROOTRIPDIR)/%.cc 
	rm -f $@ 
	@echo "Postponing $@ for separate Makefile" 

wrkdir/libRootUtils.so:
	$(MAKE) -f Makefile.rootstuff 

FitManager.o:		FitManager.cc FitManager.h wrkdir/ThrustFitManagerCUDA.o Variable.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

wrkdir/AllPdfs.o:	$(FUNCTORLIST)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(SRCDIR)/PDFs  -c src/PDFs/AllPdfs.cu -o $@ 
	@echo "$@ done"

examples:
	cd examples && $(MAKE)

clean:
	@rm -f  wrkdir/*

