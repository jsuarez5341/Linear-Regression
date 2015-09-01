data = DataSetLinearVariations; %Generates variable "data"

%Vars
m = length(data(:,1));              %Number of samples
n = length(data(1,:))-1;            %Number of Features
lbl = n+1;                          %Label column index
theta = zeros(1,n);                 %Parameter vector
a = 1;                              %Learning "constant"
eOld = 999999999999;                %Initialize errors to something high
eNew = 999999999998;

%Constants
aDiv = 2;                           %Controls reduction of alpha per overshoot
aMin = .0000001;                    %Controls execution endpoint
eMin = .0000001;                    %Controls execution endpoint
eTol = .0001;                       %Controls execution endpoint

%Debug
debugLevel = 1;                     %Used to control how much debugging is displayed
eArray = [];                        %Used to plot error graph

%---Begin algorithm execution---

while a>aMin  && abs(eNew-eOld)>eTol
    eOld = eNew;
    eNew = sum((theta*data(:, 1:n)'-data(:, lbl)').^2); %Sum squared errors
    
    if eNew>eOld
        a = a/aDiv;
    end

    if debugLevel>0
        eArray = [eArray, eNew];
    end
    
    %Update parameters
    
    %---Optimized gradient descent performed in a linear algebra grounded
    %fashion. Nearly unreadable due to vectorization: see corrosponding video for details. 
    %Note that the bsxfun call multiplies each element in a column
    %vector elementwise by a sample vector.
    theta = theta - (a/m).* sum( bsxfun(@times, (theta*data(:, 1:n)' - data(:, lbl)')' , data(:, 1:n)) );
    
end

%---End algorithm execution---

%Debug graph

if debugLevel>0
    plot(1:length(eArray), eArray), xlabel('Iteration'), ylabel('Error'), title('Linear Regression Error');
end




