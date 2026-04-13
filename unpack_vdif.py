# Written by Justin Bray (JBO)

import numpy as np
# Read header from an open VDIF file, and return a dict containing header data.
def readheader(infile):
  
  # Necessary munging of binary header to put it in the same layout as figure 3 in the format spec.
  # This was a bugger to get right.
  words = np.fromfile(infile, dtype=np.uint8, count=4*4)
  words = np.reshape(words, [4,4])[:,::-1]
  words = np.unpackbits(words, axis=1)

  header = {}

  # Parse word 0.
  header['invalid'] = words[0][0] # invalidity flag
  header['legacy'] = words[0][1] # legacy format flag
  words[0][:2] = 0 # erase flags to avoid contaminating first byte of second counter
  vals = np.packbits(words[0])
  header['seconds'] = sum(256**np.arange(3,-1,-1) * vals)

  # Parse word 1.
  assert words[1][0] == words[1][1] == 0, 'Unassigned bits should be zero.'
  header['epoch'] = np.packbits(words[1][:8])[0] # epoch in 6-month intervals from 1 Jan 2000
  vals = np.packbits(words[1][8:]) # remainder of word breaks cleanly at end of first byte
  header['framenum'] = sum(256**np.arange(2,-1,-1) * vals) # frame # within second

  # Parse word 2.
  vals = words[2][:3]
  header['version'] = sum(2**np.arange(2,-1,-1) * vals)
  assert header['version'] == 0, 'VDIF header version not supported.'
  vals = words[2][3:8] # bits in header actually represent log2(nchan)
  header['nchan'] = 2**sum(2**np.arange(4,-1,-1) * vals) # number of channels
  vals = np.packbits(words[2][8:]) # word breaks cleanly at end of first byte again
  header['framelen'] = sum(256**np.arange(2,-1,-1) * vals)*8 # frame length in bytes

  # Parse word 3.
  header['dtype'] = 'C' if words[3][0] else 'R' # complex or real data
  vals = words[3][1:6]
  header['nbits'] = sum(2**np.arange(4,-1,-1) * vals)+1 # sample size in bits
  vals = words[3][6:16]
  header['thread'] = sum(2**np.arange(9,-1,-1) * vals) # thread ID
  vals = np.packbits(words[3][16:]) # word breaks cleanly at end of second byte this time
  header['station'] = sum(256**np.arange(1,-1,-1) * vals) # station ID

  # Parse remaining words if necessary.
  if not header['legacy']:
    xwords = np.fromfile(infile, dtype=np.uint8, count=4*4)
    assert not xwords.sum(), 'Not set up to process extended user data.'

  # Derived values.
  header['headerlen'] = 16 + 16*(1 - header['legacy']) # bytes; full length of header
  header['datalen'] = header['framelen'] - header['headerlen'] # bytes; length of data only
  # Nicely-formatted dates, etc., would go here, derived from 'epoch' and 'seconds'.

  return header

# Read frames from an open VDIF file, up to the entire file, from specified starting frame.
def readframes(infile, header, nframes=99999999999, initframe=0):
  # Find file size.
  infile.seek(0,2)
  filesize = infile.tell()

  assert not filesize % header['framelen'], 'File size must be evenly divisible by frame length.'

  # Reading will start from initial frame.
  infile.seek(initframe*header['framelen'],0)
  # Do not read beyond end of file.
  nframes = min(nframes, filesize // header['framelen'] - initframe)

  raw = np.fromfile(infile, np.uint8, count=nframes*header['framelen'])
  raw = np.reshape(raw, [nframes, header['framelen']])

  # Extract parameters that vary between frames.
  # First, seconds from reference epoch.
  seconds = raw[:,3::-1]
  seconds = np.unpackbits(seconds, axis=1)
  seconds[:,:2] = 0 # blank out flags
  seconds = np.packbits(seconds, axis=1).astype(np.int64)
  seconds = 256**3*seconds[:,0] + 256**2*seconds[:,1] + 256**1*seconds[:,2] + 256**0*seconds[:,3]
  # Second, frame number within second.
  framenums = raw[:,6:3:-1].astype(np.int64)
  framenums = 256**2*framenums[:,0] + 256**1*framenums[:,1] + 256**0*framenums[:,2]
  # Third, thread ID.
  threads = raw[:,15:13:-1]
  threads = np.unpackbits(threads, axis=1)
  threads[:,:6] = 0 # blank out flag and sample size
  threads = np.packbits(threads, axis=1).astype(np.int64)
  threads = 256**1*threads[:,0] + 256**0*threads[:,1]

  framedata = raw[:,header['headerlen']:]

  return framedata, seconds, framenums, threads


# Sort frame data into separate sequences for each thread.
def sortframes(framedata, seconds, framenums, threads):
  ithreads = np.unique(threads)

  # Find separate index list for frames for each thread.
  lexinds = np.lexsort(( framenums, seconds ))
  inds = []
  for ithread in ithreads:
    tinds = lexinds[np.where(threads[lexinds] == ithread)]
    inds.append(tinds)

  # Trim mismatched preceding frames.
  fseconds   = seconds  [[tinds[0] for tinds in inds]]
  fframenums = framenums[[tinds[0] for tinds in inds]]
  while len(np.unique(fseconds)) > 1 or len(np.unique(fframenums)) > 1:
    ftind = np.lexsort(( fframenums, fseconds ))[0]
    inds[ftind] = inds[ftind][1:]
    fseconds   = seconds  [[tinds[0] for tinds in inds]]
    fframenums = framenums[[tinds[0] for tinds in inds]]

  # Trim mismatched following frames.
  fseconds   = seconds  [[tinds[-1] for tinds in inds]]
  fframenums = framenums[[tinds[-1] for tinds in inds]]
  while len(np.unique(fseconds)) > 1 or len(np.unique(fframenums)) > 1:
    ftind = np.lexsort(( fframenums, fseconds ))[-1]
    inds[ftind] = inds[ftind][:-1]
    fseconds   = seconds  [[tinds[-1] for tinds in inds]]
    fframenums = framenums[[tinds[-1] for tinds in inds]]

  for ithread,tinds in zip(ithreads,inds):
    assert max(np.diff(seconds[tinds])) <= 1, 'Missing data in thread %d.' % ithread
  # Note that this check will not notice if the missing frame is the final frame in a second.

  assert len(np.unique([len(tinds) for tinds in inds])) == 1, 'Different numbers of frames in different threads.'
  inds = np.array(inds) # we can do this now that we know the array is perfectly rectangular

  nthreads = len(ithreads)
  assert ithreads.max() == nthreads-1, 'Discontiguous thread numbering.'
  # Assumed from this point onward.

  # Copy out thread data.
  framelen = np.shape(framedata)[1]
  nframes = np.shape(inds)[1]
  threaddata = np.zeros([nthreads,nframes,framelen], dtype=framedata.dtype)

  for ithread,tinds in enumerate(inds):
    for iframe,tind in enumerate(tinds):
      threaddata[ithread,iframe,:] = framedata[tind,:]

  threaddata = np.reshape(threaddata, [len(ithreads), nframes*framelen])

  return threaddata


def unpacksamps(samps, nbits, dtype):
  assert nbits == 2, 'Unpacking procedure is hard-coded for 2-bit samples.'
  assert dtype == 'R', 'Unpacking procedure assumes real samples.'
  assert len(np.shape(samps)) == 1, 'Unpacking procedure is written for one-dimensional arrays.'

  nmax = 2**24 # max bytes to unpack at a time

  if len(samps) <= nmax:
    samps = np.unpackbits(samps)
    samps = 2*samps[0::2] + samps[1::2]

    # Reorder 1-byte chunks.
    samps = np.reshape(samps, [int(len(samps)/4), 4])
    samps = samps[:,::-1]
    samps = np.reshape(samps, np.size(samps))

    return samps
  
   # We have a large amount of data.  Do it a chunk at a time.

  oldsamps = samps
  samps = np.zeros(len(oldsamps)*4, dtype=np.uint8)
  isamp = 0

  while len(oldsamps):
    tmpsamps = oldsamps[:nmax]
    oldsamps = oldsamps[nmax:]

    tmpsamps = unpacksamps(tmpsamps, nbits, dtype)

    samps[isamp:isamp+len(tmpsamps)] = tmpsamps
    isamp += len(tmpsamps)

  return samps

