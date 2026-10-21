import base64,hashlib,pathlib,re,sys
bad={'d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4','86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b'}
pattern=re.compile(rb'-----BEGIN (?:TRUSTED )?CERTIFICATE-----\s*(.*?)\s*-----END (?:TRUSTED )?CERTIFICATE-----',re.S)
def fingerprint(m):
 der=base64.b64decode(re.sub(rb'\s+',b'',m.group(1)));length=der[1];header=2
 if length&128:
  count=length&127;length=int.from_bytes(der[2:2+count],'big');header+=count
 assert der[0]==48 and header+length<=len(der)
 return hashlib.sha256(der[:header+length]).hexdigest()
for arg in sys.argv[1:]:
 root=pathlib.Path(arg)/'etc/ssl/certs'
 if not root.exists():continue
 for path in root.iterdir():
  if path.is_symlink() or not path.is_file():continue
  data=path.read_bytes();filtered=pattern.sub(lambda m:b'' if fingerprint(m) in bad else m.group(0),data)
  if filtered==data:continue
  if pattern.search(filtered):path.write_bytes(filtered)
  else:path.unlink()
 for path in root.iterdir():
  if path.is_symlink() and not path.exists():path.unlink()
 for path in root.iterdir():
  if path.is_file():assert not any(fingerprint(m) in bad for m in pattern.finditer(path.read_bytes())),path
