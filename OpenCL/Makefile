# OpenCL SDK Version 1.00.0419.1830

PROJECTS := $(shell find src -name Makefile)

ifeq ($(dbg),1)
	BINDIR := bin/linux/debug/
else 
    BINDIR := bin/linux/release/
endif


%.ph_build : common/liboclUtil.so shared/libshrutil.so 
	make -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean : 
	make -C $(dir $*) clean $(USE_DEVICE)

%.ph_clobber :
	make -C $(dir $*) clobber $(USE_DEVICE)

all:  $(addsuffix .ph_build,$(PROJECTS))
	@echo "Finished building all"

common/liboclUtil.so:
	@make -C common

shared/libshrutil.so:
	@make -C ../shared/ 

tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))
	@make -C common clean

clobber: clean $(addsuffix .ph_clobber,$(PROJECTS))
	@make -C common clobber

runall: 
	$(SHELL) common/runall.sh $(BINDIR)
