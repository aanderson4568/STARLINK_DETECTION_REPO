% make_lte_waveform.m -- generate a standard LTE downlink with MATLAB's LTE Toolbox and save
% it in the format detectiontest1's Python scripts read (.cf32 + .json sidecar).
%
% Needs: LTE Toolbox (check with `ver lte`).
% Run it from this folder; it writes ../out/lte_matlab.cf32 and ../out/lte_matlab.json.
% Then in lte_test.py set  LTE_FILE = "out/lte_matlab.cf32"  and run  python lte_test.py

pci        = 378;       % cell ID (0-503) -- lte_test.py checks it decodes back to this
subframes  = 40;        % 40 ms; keep it a multiple of 10 so the Pluto's loop is seamless
centerFreq = 1995e6;    % only written into the .json; the Pluto is tuned by lte_test.py

% R.6 = 5 MHz reference measurement channel: 25 resource blocks, 1 antenna port.
% Unlike a plain test pattern it has real coded MIB (PBCH), PCFICH, PDCCH and PDSCH.
rmc = lteRMCDL('R.6');
rmc.NCellID = pci;
rmc.TotSubframes = subframes;

data = randi([0 1], 1e6, 1);          % user data bits (reused as needed)
[waveform, ~, info] = lteRMCDLTool(rmc, data);

fs = info.SamplingRate;               % 7.68e6 for 5 MHz
fprintf('Generated %d samples (%.1f ms) at %.2f Msps, PCI %d\n', ...
        size(waveform, 1), 1e3 * size(waveform, 1) / fs, fs / 1e6, pci);

% Scale so the biggest I or Q value is 0.9 -- the Pluto clips above 1.0.
w = waveform(:, 1);
w = w / max(abs([real(w); imag(w)])) * 0.9;

% .cf32 = interleaved I, Q, I, Q ... as 32-bit floats (little-endian on Windows).
outDir = fullfile(fileparts(mfilename('fullpath')), '..', 'out');
if ~exist(outDir, 'dir'), mkdir(outDir); end
iq = zeros(2 * numel(w), 1, 'single');
iq(1:2:end) = real(w);
iq(2:2:end) = imag(w);
fid = fopen(fullfile(outDir, 'lte_matlab.cf32'), 'w');
fwrite(fid, iq, 'single');
fclose(fid);

meta = struct('sample_rate_hz', fs, 'center_frequency_hz', centerFreq, ...
              'n_samples', numel(w), 'pci', pci, 'source', 'MATLAB lteRMCDLTool R.6');
fid = fopen(fullfile(outDir, 'lte_matlab.json'), 'w');
fprintf(fid, '%s', jsonencode(meta, 'PrettyPrint', true));
fclose(fid);
fprintf('Saved %s\n', fullfile(outDir, 'lte_matlab.cf32'));
