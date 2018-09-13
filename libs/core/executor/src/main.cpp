#include <iostream>
#include <string>
#include <functional>
#include <unordered_map>
#include <memory>
#include <queue>

typedef int EventData;
typedef std::function<void(EventData)> EventHandler;
typedef std::list<const EventHandler>::const_iterator EventId;

class EventSource
{
public:
    virtual ~EventSource() {}
    virtual EventId addEventListener(std::string name, EventHandler handler) = 0;
    virtual void removeEventListener(std::string name, EventId id) = 0;
};

class EventTarget
{
public:
    virtual ~EventTarget() {}
    virtual void raiseEvent(std::string name, EventData data) = 0;
};

class EventManager : public EventSource, public EventTarget
{
private:
    std::unordered_map<std::string, std::list<const EventHandler>> event_handlers;
public:
    EventManager() = default;
    EventManager(EventManager const &) = delete;
    void operator=(EventManager const &) = delete;

    EventId addEventListener(std::string name, EventHandler handler) final
    {
        auto &current_handlers = event_handlers[name];
        return current_handlers.insert(current_handlers.end(), handler);
    }

    void removeEventListener(std::string name, EventId id) final
    {
        auto &current_handlers = event_handlers[name];
        current_handlers.erase(id);
    }

    void raiseEvent(std::string name, EventData data) final
    {
        for (auto const &handler : event_handlers[name])
        {
            handler(data);
        }
    }
};

#define IMPLEMENT_EVENT_SOURCE(event_source) \
    EventId addEventListener(std::string event_name, EventHandler event_handler) final { return (event_source).addEventListener(event_name, event_handler); } \
    void removeEventListener(std::string event_name, EventId event_id) final { return (event_source).removeEventListener(event_name, event_id); }

#define IMPLEMENT_EVENT_TARGET(event_target) \
    void raiseEvent(std::string event_name, EventData event_data) final { (event_target).raiseEvent(event_name, event_data); }


typedef int SignalData;

class SignalSource
{
public:
    virtual std::unique_ptr<SignalData> get_signal(std::string name) = 0;
};

class SignalTarget
{
public:
    virtual void signal(std::string name, SignalData data) = 0;
};

class SignalManager : public SignalTarget, public SignalSource
{
private:
    //int next_signal_id = 0;
    std::unordered_map<std::string, std::queue<SignalData>> unhandled_signals;
public:
    void signal(std::string name, SignalData data) final
    {
        unhandled_signals[name].push(data);
    }

    std::unique_ptr<SignalData> get_signal(std::string name) final
    {
        auto &queue = unhandled_signals[name];
        
        if (queue.empty())
            return nullptr;

        auto ret = std::make_unique<SignalData>(queue.front());
        queue.pop();
        return ret;
    }
};

#define IMPLEMENT_SIGNAL_SOURCE(signal_source) \
    std::unique_ptr<SignalData> get_signal(std::string signal_name) { return (signal_source).get_signal(signal_name); }

#define IMPLEMENT_SIGNAL_TARGET(signal_target) \
    void signal(std::string signal_name, SignalData signal_data) { (signal_target).signal(signal_name, signal_data); }

class Executor : public EventSource, protected EventTarget, protected SignalSource, public SignalTarget
{
private:
    EventManager event_manager;
    SignalManager signal_manager;
protected:
    IMPLEMENT_EVENT_TARGET(event_manager)
    IMPLEMENT_SIGNAL_SOURCE(signal_manager)
public:
    IMPLEMENT_EVENT_SOURCE(event_manager)
    IMPLEMENT_SIGNAL_TARGET(signal_manager)

    virtual void execute() = 0;
};

class XExecutor : public Executor
{
public:
    void execute() final;
};

void XExecutor::execute()
{
    int sum = 0;

    raiseEvent("updated", sum);

    while(true)
    {
        auto end_signal = get_signal("end");

        if (end_signal) break;

        auto add_signal = get_signal("add");

        if(add_signal)
        {
            const int value = *add_signal;
            sum += value;
            raiseEvent("updated", sum);
        }
    }

    raiseEvent("end", sum);
}

int main()
{
    std::cout << "Creating...\n";

    XExecutor e;

    e.addEventListener("updated", [&e](int value)
    {
        std::cout << "Event \"updated\" with param \"" << value << "\".\n";
        
        if (value < 40)
        {
            e.signal("add", value+1);
        }
        else
        {
            e.signal("end", 0);
        }
    });

    e.addEventListener("end", [&e](int value)
    {
        std::cout << "Event \"end\" with param \"" << value << "\".\n";
    });

    e.execute();

    std::cout << "Done.\n";
}