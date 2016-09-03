% this script is adapted from
% https://gist.github.com/karpathy/d4dee566867f8291f086
function it()
  inputText = 'hello world! ';
  nHiddenStates = 128;
  nSeqLength = 8;

  % http://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
  %{
  inputText = fileread('t8.shakespeare.txt');
  nHiddenStates = 256;
  nSeqLength = 32;
  %}

  % repeat the text if it's too short
  while length(inputText) < nSeqLength
    inputText = [inputText inputText];
  end

  % we need vectors in rows
  inputText = inputText';
  chars = unique(inputText);
  indices = [1:length(chars)]';

  % inputs are indices of chars, which can be obtained from inputText
  inputs = inputText * 0;
  for t = 1:length(chars)
    inputs(inputText == chars(t)) = t;
  end
  nUniqChars = length(chars);

  % network parameters
  Wxh = rand(nHiddenStates, nUniqChars) * 0.1;
  Whh = rand(nHiddenStates, nHiddenStates) * 0.1;
  Why = rand(nUniqChars, nHiddenStates) * 0.1;
  bh = zeros(nHiddenStates, 1);
  by = zeros(nUniqChars, 1);

  function [map loss hprev] = calcLoss(inputs, outputs, prevHiddenLayer)
    map = containers.Map('KeyType', 'double', 'ValueType', 'any');

    loss = 0;
    map(0) = struct('h', prevHiddenLayer);
    hprev = 0;
    for t = 1:length(inputs)
      x = [indices == inputs(t)];
      h = tanh(Wxh * x + Whh * map(t - 1).h + bh);
      y = Why * h + by;

      p = exp(y) / sum(exp(y));
      loss = loss - log(p(outputs(t)));

      map(t) = struct('x', x, 'h', h, 'o', y, 'p', p);
      hprev = h;
    end
  end

  function [ dWxh dWhh dWhy dbh dby ] = calcDiff(inputs, outputs, map)
    dWxh = Wxh * 0;
    dWhh = Whh * 0;
    dWhy = Why * 0;
    dbh = bh * 0;
    dby = by * 0;
    dNextHiddenLayer = 0;
    for t = length(inputs):-1:1
      dy = map(t).p - [indices == outputs(t)];
      dWhy = clamp(dWhy + tensor(dy, map(t).h));
      dby  = clamp(dby + dy);
      dh   = clamp(Why' * dy + dNextHiddenLayer);

      dhraw = (1 - map(t).h .* map(t).h) .* dh;
      dbh  = clamp(dbh + dhraw);
      dWxh = clamp(dWxh + tensor(dhraw, map(t).x));
      dWhh = clamp(dWhh + tensor(dhraw, map(t - 1).h));

      dNextHiddenLayer = Whh' * dhraw;
    end
  end

  % http://sebastianruder.com/optimizing-gradient-descent/index.html
  % we use RMSprop here
  mWxh = Wxh * 0;
  mWhh = Whh * 0;
  mWhy = Why * 0;
  mbh = bh * 0;
  mby = by * 0;

  learningRate = 0.001;
  adagradTerm = 1e-8;
  RMSPropFac = 0.9;
  function RMSpropUpdate(dWxh, dWhh, dWhy, dbh, dby)
    mWxh = RMSPropFac * mWxh + (1 - RMSPropFac) * dWxh .* dWxh;
    Wxh = Wxh - learningRate * dWxh ./ sqrt(mWxh + adagradTerm);
    mWhh = RMSPropFac * mWhh + (1 - RMSPropFac) * dWhh .* dWhh;
    Whh = Whh - learningRate * dWhh ./ sqrt(mWhh + adagradTerm);
    mWhy = RMSPropFac * mWhy + (1 - RMSPropFac) * dWhy .* dWhy;
    Why = Why - learningRate * dWhy ./ sqrt(mWhy + adagradTerm);
    mbh  = RMSPropFac * mbh  + (1 - RMSPropFac) * dbh .* dbh;
    bh  = bh - learningRate  * dbh ./ sqrt(mbh + adagradTerm);
    mby  = RMSPropFac * mby  + (1 - RMSPropFac) * dby .* dby;
    by  = by - learningRate  * dby ./ sqrt(mby + adagradTerm);
  end

  function v = sampleOutput(h, c, nOutput)
    v = [ ];
    for t = 1:nOutput
      x = [chars == c];
      h = tanh(Wxh * x + Whh * h + bh);
      y = Why * h + by;
      %c = chars(y == max(y));
      p = exp(y) / sum(exp(y));
      c = randsample(chars, 1, true, p);

      v(t) = c;
    end
  end

  mLoss = [ log(nUniqChars) ];
  mBytes = [ 0 ];
  function plotLoss(offset, loss)
    mBytes(end + 1) = offset / 1e3;
    mLoss(end + 1) = loss / nSeqLength;

    if length(mLoss) > 1000
      mBytes = mBytes(end - 800:end);
      mLoss = mLoss(end - 800:end);
    end

    plot(mBytes, mLoss);
    xlabel('size (kB)');
    ylabel('loss (a.u.)')
    drawnow;
  end

  count = 0;
  offset = 0;
  sampleInputs = [inputs inputs];
  prevHiddenLayer = bh * 0;
  while count < 1e5
    offsets = (1:nSeqLength) + mod(offset, length(inputText)) + 1;
    offsetInputs = sampleInputs(offsets);
    offsetOutputs = sampleInputs(offsets + 1);

    [ map loss hprev ] = calcLoss(offsetInputs, offsetOutputs, prevHiddenLayer);
    [ dWxh dWhh dWhy dbh dby ] = calcDiff(offsetInputs, offsetOutputs, map);

    prevHiddenLayer = hprev;
    RMSpropUpdate(dWxh, dWhh, dWhy, dbh, dby);

    if mod(count, 10) == 0
      plotLoss(offset, loss);
    end

    if mod(count, 100) == 0
      output = sampleOutput(prevHiddenLayer, chars(offsetInputs(1)), 200);
      disp(sprintf('[loss %g from latest %d bytes]\n%s', loss / nSeqLength, offset, output));
    end

    offset = offset + nSeqLength;
    count = count + 1;
  end

end

function y = clamp(val)
  y = min(max(val, -5), 5);
end

function x = tensor(a, b)
  [m n] = meshgrid(b, a);
  x = m .* n;
end
