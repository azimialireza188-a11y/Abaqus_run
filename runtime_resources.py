"""All available resources; no fixed capacity reservation or savings ceiling.

This module is stdlib-only so it can configure BLAS before NumPy is imported.
"""
from dataclasses import dataclass, asdict
import ctypes
import inspect
import os
import subprocess


@dataclass
class ResourceInventory:
    logical_cpus: int
    physical_cores: int
    total_memory_bytes: int
    available_memory_bytes: int
    devices: list


@dataclass
class ResourcePolicy:
    cpus: int
    gpus: int
    inventory: ResourceInventory
    memory_reserve_bytes: int = 0
    gpu_reserve_bytes: int = 0

    def provenance(self):
        return asdict(self)


def available_memory():
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (k, ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page',
                                                 'avail_page','total_virtual','avail_virtual','avail_extended')]
        s = MemoryStatus(); s.length = ctypes.sizeof(s)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            raise OSError('Cannot query physical memory')
        return int(s.total_phys), int(s.avail_phys)
    if os.path.exists('/proc/meminfo'):
        with open('/proc/meminfo') as f:
            entries = {l.split(':')[0]: int(l.split()[1])*1024 for l in f if l.split()[1].isdigit()}
        total, available = entries['MemTotal'], entries.get('MemAvailable', entries['MemFree'])
        try:
            with open('/sys/fs/cgroup/memory.max') as f: maximum=f.read().strip()
            with open('/sys/fs/cgroup/memory.current') as f: current=int(f.read())
            if maximum != 'max':
                total=min(total,int(maximum)); available=min(available,max(0,int(maximum)-current))
        except (OSError, ValueError):
            pass
        return total, available
    raise OSError('Memory discovery unsupported on this platform')


def detect_resources():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    physical = cpus
    try:
        if os.name == 'nt':
            physical = int(subprocess.check_output(['powershell','-NoProfile','-Command',
                '(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum'],
                text=True, timeout=10).strip())
        elif os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo') as f: sections=f.read().strip().split('\n\n')
            ids=set()
            for section in sections:
                items=dict(l.split(':',1) for l in section.splitlines() if ':' in l)
                ids.add((items.get('physical id\t','0').strip(),items.get('core id\t',items.get('processor\t','0')).strip()))
            physical=min(cpus,len(ids)) or cpus
    except (OSError, ValueError, subprocess.SubprocessError):
        physical=cpus
    total, available=available_memory()
    devices=[]
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=5, stderr=subprocess.DEVNULL)
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        selected=None if visible is None else {v.strip() for v in visible.split(',') if v.strip()}
        for line in output.strip().splitlines():
            i,uuid,name,free=[s.strip() for s in line.split(',',3)]
            if selected is None or i in selected or any(uuid.startswith(v) for v in selected):
                devices.append(dict(id=i,uuid=uuid,name=name,free_bytes=int(free)*1024**2))
    except (OSError,ValueError,subprocess.SubprocessError):
        pass
    return ResourceInventory(cpus,physical,total,available,devices)


def resolve_policy(inventory, cpu_override=None, gpu_override=None):
    cpus=inventory.logical_cpus if cpu_override in (None,'auto') else int(cpu_override)
    gpus=len(inventory.devices) if gpu_override in (None,'auto') else int(gpu_override)
    if cpus<1 or gpus<0 or gpus>len(inventory.devices):
        raise ValueError('Invalid CPU/GPU request for visible devices')
    return ResourcePolicy(cpus,gpus,inventory)


def resolve_solver_policy(inventory, cpu_override=None, gpu_override=None):
    """Abaqus solver CPU availability can count cores rather than SMT threads.

    Use every physical core by default; keep the logical CPU policy for Python
    postprocessing. This imposes no reserve or fixed worker/memory ceiling.
    """
    solver_cpus = inventory.physical_cores if cpu_override in (None,'auto') else cpu_override
    return resolve_policy(inventory, solver_cpus, gpu_override)


def batch_capacity(available, bytes_per_item, fixed_bytes=0):
    if bytes_per_item<=0 or fixed_bytes<0:
        raise ValueError('Invalid batch working-set estimate')
    n=(int(available)-int(fixed_bytes))//int(bytes_per_item)
    if n<1:
        raise MemoryError('Live available memory cannot fit one item')
    return n


def configure_threads(cpus):
    if cpus<1: raise ValueError('Thread count must be positive')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]=str(cpus)
    os.environ['OMP_DYNAMIC']='FALSE'
    os.environ['MKL_DYNAMIC']='FALSE'


def supports_gpu_keyword(factory):
    try:
        return 'numGPUs' in inspect.signature(factory).parameters
    except (TypeError,ValueError):
        return 'numGPUs' in (getattr(factory,'__doc__','') or '')


def abaqus_job_settings(policy, capabilities):
    result=dict(numCpus=policy.cpus,numDomains=policy.cpus,memory=100,getMemoryFromAnalysis=False)
    if policy.gpus and capabilities.get('numGPUs'):
        result['numGPUs']=policy.gpus
    return result


def process_memory():
    """Observed current-process RSS/peak, separate from resource requests."""
    import sys
    if os.name=='nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        getprocess=ctypes.windll.kernel32.GetCurrentProcess
        getprocess.restype=wintypes.HANDLE
        fn=ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not fn(getprocess(),ctypes.byref(counters),counters.cb):raise OSError('Cannot query process memory')
        current=int(counters.WorkingSetSize);peak=int(counters.PeakWorkingSetSize)
    else:
        import resource
        peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
        current=None
        if os.path.exists('/proc/self/status'):
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):current=int(line.split()[1])*1024
                    if line.startswith('VmHWM:'):peak=max(peak,int(line.split()[1])*1024)
    if current is not None:peak=max(peak,current)
    return dict(current_rss_bytes=current,peak_rss_bytes=peak,scope='CURRENT_PROCESS_LIFETIME')

