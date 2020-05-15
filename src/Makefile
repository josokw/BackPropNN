CXX = g++ 
CXXFLAGS = -std=c++2a -g -Wall -Wextra -Weffc++ -Wpedantic \
           -Wold-style-cast -Winit-self -Wno-unused -Wshadow \
           -Wno-parentheses -Wlogical-op -Wredundant-decls \
           -Wcast-align -Wsign-promo -Wmissing-include-dirs \
           -Woverloaded-virtual -Wctor-dtor-privacy

EXECUTABLE = backpropnn
SOURCES = ${wildcard *.cpp}
HEADERS = ${wildcard *.h}
OBJECTS = ${SOURCES:.cpp=.o}

.PHONY: all
all: ${EXECUTABLE}

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDXXFLAGS) -o $@

# Create dependency file
depend: $(SOURCES)
	$(CXX) $(CXXFLAGS) -MM  $^ > $@

-include depend

# Create a clean environment
.PHONY: clean
clean:
	$(RM) $(EXECUTABLE) $(OBJECTS)

# Clean up dependency file  
.PHONY: clean-depend
clean-depend: clean
	$(RM) depend  
	                                              
