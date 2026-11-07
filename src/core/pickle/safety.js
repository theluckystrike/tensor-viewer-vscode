'use strict';
// Rates every name a pickle imports or calls. Python's pickle.load imports each GLOBAL / STACK_GLOBAL
// target and calls each REDUCE / NEWOBJ / INST / OBJ target, so these names say what loading would do.

const norm = q => q.replace(/^__builtin__\./, 'builtins.').replace(/^copy_reg\./, 'copyreg.').replace(/^numpy\._core\./, 'numpy.core.');

// Modules whose import or call can run commands, run code, touch files or the network.
const DANGER_MODULES = [
  'os', 'posix', 'nt', 'subprocess', 'pty', 'sys', 'importlib', 'runpy', 'shutil', 'socket', 'ctypes', 'code', 'codeop',
  'commands', 'popen2', 'webbrowser', 'pickle', 'cPickle', '_pickle', 'marshal', 'multiprocessing', 'asyncio', 'http',
  'urllib', 'urllib2', 'urllib3', 'requests', 'ftplib', 'smtplib', 'telnetlib', 'tempfile', 'pip', 'setuptools', 'distutils',
  'bdb', 'pdb', 'timeit', 'trace', 'profile', 'cProfile', 'imp', 'zipimport', 'signal', 'shelve', 'dill', 'cloudpickle',
  'joblib', 'torch.hub', 'torch.load', 'yaml', 'xmlrpc', 'paramiko', 'fabric', 'docker', 'boto3', 'builtins.__import__',
];
const DANGER_BUILTINS = new Set([
  'eval', 'exec', 'compile', 'open', '__import__', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars', 'input',
  'breakpoint', 'execfile', 'file', 'reload', 'apply', 'help', 'exit', 'quit', 'memoryview', 'super', 'type', 'classmethod', 'staticmethod', 'property',
]);
const DANGER_EXACT = new Set([
  'platform.popen', 'types.FunctionType', 'types.CodeType', 'types.LambdaType', 'types.MethodType', 'types.ModuleType',
  'functools.partial', 'functools.reduce', 'operator.attrgetter', 'operator.methodcaller', 'operator.itemgetter', 'operator.getitem',
  'operator.call', 'numpy.load', 'numpy.lib.npyio.load', 'numpy.fromfile', 'numpy.core.multiarray.fromfile', 'numpy.testing._private.utils.runstring',
  'torch.load', 'torch.serialization.load', 'torch._C._load', 'pandas.read_pickle', 'pandas.io.pickle.read_pickle', 'logging.config.listen',
  'site.main', 'builtins.object.__reduce__', 'copyreg.__reduce_ex__', 'builtins.getattr',
]);

// Names that rebuild plain data and are fine to import.
const KNOWN_DATA = [
  /^builtins\.(set|frozenset|bytearray|bytes|complex|slice|range|object|dict|list|tuple|int|float|str|bool|NoneType|Ellipsis|NotImplemented)$/,
  /^copyreg\.(_reconstructor|__newobj__|__newobj_ex__)$/, /^_codecs\.encode$/,
  /^collections\.(OrderedDict|defaultdict|Counter|deque|namedtuple)$/, /^datetime\.(datetime|date|time|timedelta|timezone)$/,
  /^decimal\.Decimal$/, /^fractions\.Fraction$/, /^uuid\.UUID$/, /^pathlib\.(PosixPath|WindowsPath|PurePosixPath|PureWindowsPath|Path)$/,
  /^numpy\.(dtype|ndarray|core\.multiarray\.(_reconstruct|scalar)|core\.numeric\._frombuffer|ma\.core\.(_mareconstruct|MaskedArray)|random\._pickle\.\w+|matrix|recarray|core\.records\.recarray)$/,
  /^numpy\.(bool_|int8|int16|int32|int64|uint8|uint16|uint32|uint64|float16|float32|float64|complex64|complex128|str_|bytes_)$/,
  /^pandas\.(core\.(frame\.DataFrame|series\.Series|indexes\.\w+\.\w+|internals\.\w+\.\w+|arrays\.\w+\.\w+|dtypes\.dtypes\.\w+)|_libs\.(internals\._unpickle_block|tslibs\.\w+\.\w+|lib\.\w+|arrays\.\w+)|Timestamp|Timedelta)$/,
  /^sklearn\./, /^scipy\.sparse\./, /^torch\._utils\._rebuild_\w+$/, /^torch\.(FloatStorage|DoubleStorage|HalfStorage|LongStorage|IntStorage|ByteStorage|BoolStorage|BFloat16Storage|storage\._load_from_bytes|Size|float32|float16|bfloat16|int64)$/,
  /^collections\.abc\./, /^enum\./, /^re\._compile$/, /^builtins\.(Exception|ValueError|KeyError|TypeError)$/,
];

/**
 * @returns {{level:'danger'|'warn'|'data'|'custom', why:string}}
 * danger = loading would run a command, run code, read or write files or use the network.
 * warn = can be chained into code execution (getattr-like helpers).
 * data = a well-known data type.
 * custom = any other module: Python imports it and runs its top-level code when the file loads.
 */
function rate(target) {
  const q = norm(String(target || ''));
  const [mod] = splitName(q);
  if (DANGER_EXACT.has(q)) {
    if (/^(functools|operator)\./.test(q)) return { level: 'warn', why: 'A helper that pickle exploits chain together to reach code. Fine in some files, but check where it points.' };
    return { level: 'danger', why: 'Loading this file would call it. It can run code or read files.' };
  }
  if (mod === 'builtins') {
    const name = q.slice('builtins.'.length);
    if (DANGER_BUILTINS.has(name)) return { level: 'danger', why: `builtins.${name} can run code or reach any object. Loading this file would call it.` };
  }
  for (const m of DANGER_MODULES) {
    if (q === m || q.startsWith(m + '.')) return { level: 'danger', why: dangerWhy(m) };
  }
  if (KNOWN_DATA.some(re => re.test(q))) return { level: 'data', why: 'A known data type.' };
  if (mod === 'builtins') return { level: 'warn', why: 'A Python built-in that is not a plain data type.' };
  if (mod === '__main__') return { level: 'custom', why: 'A class from the script that saved the file. Python needs that script to load it.' };
  return { level: 'custom', why: `Python imports ${mod} and runs its module code when it loads this file. Check that you trust that package.` };
}

function dangerWhy(m) {
  if (['os', 'posix', 'nt', 'subprocess', 'pty', 'commands', 'popen2', 'platform'].includes(m)) return 'Runs shell commands or programs. Loading this file would run it.';
  if (['socket', 'http', 'urllib', 'urllib2', 'urllib3', 'requests', 'ftplib', 'smtplib', 'telnetlib', 'webbrowser', 'xmlrpc', 'paramiko', 'boto3', 'docker', 'fabric'].includes(m)) return 'Uses the network. Loading this file would call it.';
  if (['shutil', 'tempfile', 'shelve'].includes(m)) return 'Reads, writes or deletes files. Loading this file would call it.';
  if (['pickle', 'cPickle', '_pickle', 'marshal', 'dill', 'cloudpickle', 'joblib', 'yaml', 'torch.load', 'torch.hub'].includes(m)) return 'Loads more serialized data, which can hide a second payload.';
  return 'Can run code. Loading this file would call it.';
}

function splitName(q) {
  const i = q.indexOf('.');
  return i < 0 ? [q, ''] : [q.slice(0, i), q.slice(i + 1)];
}

/**
 * Summarise the recorded events. Each GLOBAL/STACK_GLOBAL/INST target and each REDUCE/NEWOBJ/OBJ/BUILD target is rated.
 * argText(value) turns an argument tuple into short text for the table.
 */
function analyze(events, eventCount, argText) {
  const rows = events.map(e => {
    const r = /object$|^(None|list|tuple|dict|set|frozenset|string|number|boolean|bigint|numpy\.dtype|numpy\.ndarray)$/.test(e.target) && e.op !== 'GLOBAL' && e.op !== 'STACK_GLOBAL' && e.op !== 'INST'
      ? { level: 'data', why: e.op === 'BUILD' ? 'Sets the state of an object made earlier.' : 'Calls a value made earlier in the file.' }
      : rate(e.target);
    if (/ object$/.test(e.target)) {
      const base = rate(e.target.replace(/ object$/, ''));
      if (base.level !== 'data') Object.assign(r, base);
    }
    return { op: e.op, pos: e.pos, target: e.target, level: r.level, why: r.why, args: e.args !== undefined && argText ? argText(e.args) : undefined };
  });
  const imports = new Map();
  for (const r of rows) {
    if (r.op !== 'GLOBAL' && r.op !== 'STACK_GLOBAL' && r.op !== 'INST') continue;
    const k = norm(r.target);
    if (!imports.has(k)) imports.set(k, { target: r.target, level: r.level, why: r.why, count: 0, first: r.pos });
    imports.get(k).count++;
  }
  const calls = rows.filter(r => r.op === 'REDUCE' || r.op === 'NEWOBJ' || r.op === 'NEWOBJ_EX' || r.op === 'OBJ' || r.op === 'INST');
  const danger = [...imports.values()].filter(i => i.level === 'danger');
  const warn = [...imports.values()].filter(i => i.level === 'warn');
  const custom = [...imports.values()].filter(i => i.level === 'custom');
  let verdict;
  if (danger.length) {
    const first = rows.find(r => r.level === 'danger' && r.op === 'REDUCE') || rows.find(r => r.level === 'danger');
    verdict = { level: 'danger', title: `Risky: loading this file in Python would call ${danger.map(d => d.target).slice(0, 3).join(', ')}${danger.length > 3 ? ` and ${danger.length - 3} more` : ''}.`, example: first && first.args ? `${first.target}${first.args}` : undefined };
  } else if (warn.length) {
    verdict = { level: 'warn', title: `Check before loading: ${warn.map(d => d.target).slice(0, 3).join(', ')} can be used to reach code.` };
  } else if (custom.length) {
    verdict = { level: 'custom', title: `No risky names found. Loading needs ${custom.length} module${custom.length === 1 ? '' : 's'} outside the standard data types: ${custom.map(c => c.target.split('.')[0]).filter((v, i, a) => a.indexOf(v) === i).slice(0, 4).join(', ')}.` };
  } else {
    verdict = { level: 'data', title: imports.size ? `No risky names found. All ${imports.size} imported names are known data types.` : 'No imports at all. This file holds plain Python data.' };
  }
  return { verdict, rows, imports: [...imports.values()], calls: calls.length, eventCount, shown: rows.length };
}

module.exports = { rate, analyze, norm };
