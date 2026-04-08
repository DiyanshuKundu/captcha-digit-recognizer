tic
data = importdata('Train/labels.txt');
img_nrs = data(:,1);
true_labels = data(:,(2:end));

my_labels = zeros(size(true_labels));
N = numel(img_nrs);

for n = 801:1200
    k = img_nrs(n);
    %disp(n);
    im = imread(sprintf('Train/captcha_%04d.png', k));
    my_labels(k,:) = myclassifier(im);
end

correct = sum(abs(true_labels - my_labels),2)==0;
fprintf('\n\nAccuracy: \n');
fprintf('%f\n\n',mean(correct));
toc
